import os
from glob import glob
import numpy as np
from PIL import Image

from src.extraction.model import Model
from src.extraction.preprocess import Preprocessing
from src.search.indexing import KDTree
from src.search.search import KDTreeSearch, Measure

# Location of storage
GALLERY_STORAGE = os.path.join('storage', 'gallery')
EMBEDDINGS_STORAGE = os.path.join('storage', 'embedding')
ACCESS_LOGS_STORAGE = os.path.join('storage', 'access_logs')

class Pipeline:
    """
    Pipeline class for handling the extraction of embeddings, saving them, precomputing
    the KD-Tree for the gallery, and searching the gallery for nearest neighbors.
    """

    def __init__(self, preprocessing, model: Model, index, search):
        """
        Initialize the Pipeline class.
        
        Args:
            preprocessing (Preprocessing): Preprocessing object for image preprocessing.
            model (Model): Model object for extracting embeddings.
            index (KDTree): KDTree object for indexing embeddings.
            search (KDTreeSearch): KDTreeSearch object for searching the KDTree.
        """
        self.preprocessing = preprocessing
        self.model = model
        model_base_name = os.path.basename(model.model_path)
        self.model_name = os.path.splitext(model_base_name)[0]
        self.index = index
        self.search = search

    def __predict(self, probe):
        """
        Extract the embedding vector output from a preprocessed image.

        Args:
            probe (str): a location to the image file

        Returns:
            np.ndarray: The embedding vector.
        """
        # Open the image file
        probe = Image.open(probe)
        # Preprocess the image
        preprocessed_image = self.preprocessing.process(probe)
        # Extract the embedding vector
        embedding = self.model.extract(preprocessed_image)
        return embedding

    def __save_embeddings(self, filename, embedding):
        """
        Store the embeddings in a numpy format.

        Args:
            filename (str): The filename to save the embedding.
            embedding (np.ndarray): The embedding vector.
        """
        # Extract the directory from the filename and create it if it doesn't exist
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the embedding
        np.save(filename, embedding)

    def __precompute(self):
        """
        Automatically precompute the embeddings for all images in storage/gallery/*/*.jpg
        and construct a K-D Tree to organize the embeddings.
        """
        points = []
        metadata_list = []

        # Iterate over all images in the gallery
        for image_path in glob(os.path.join(GALLERY_STORAGE, "*/*.jpg")):
            # Extract first name, last name, and image ID from the file path
            _, subdir, name, image_filename = image_path.split(os.path.sep)
            name_parts = name.split('_')
            first_name = name_parts[0]
            last_name = '_'.join(name_parts[1:])
            image_id = os.path.splitext(image_filename)[0]

            # Preprocess the image and extract embeddings
            embedding = self.__predict(image_path)

            # Save the embeddings
            embedding_filename = os.path.join(EMBEDDINGS_STORAGE, self.model_name, f"{first_name}_{last_name}", f"{image_id}.npy")
            self.__save_embeddings(embedding_filename, embedding)

            # Append the embedding and metadata to the lists
            points.append(embedding)
            metadata_list.append({"first_name": first_name, "last_name": last_name, "image_filename": image_filename})

        self.points = points
        self.metadata = metadata_list
        # Construct the KD-Tree
        self.index = KDTree(k=256, points=points, metadata_list=metadata_list)
        # Update the search tree using the new index
        self.search = KDTreeSearch(self.index, self.search.distance_func)

    def search_gallery(self, probe):
        """
        Return the nearest neighbors of a probe.

        Args:
            probe (str): The location of the probe image.

        Returns:
            list: A list of dictionaries with the individual's first name, last name, and image filename.
        """
        # Extract the embedding for the probe image
        embedding = self.__predict(probe)
        # Find the nearest neighbors in the KD-Tree
        neighbors = self.search.find_nearest_neighbors(embedding, k=5)
        if not neighbors:
            return []
        # Prepare the results
        results = [{"first_name": neighbor[1]["first_name"], "last_name": neighbor[1]["last_name"], "image_filename": neighbor[1]["image_filename"]} for neighbor in neighbors]
        return results

    def precompute(self):
        """
        Public method to call the private __precompute method.
        """
        self.__precompute()

    def predict(self, image_path):
        """
        Public method to extract the embedding of an image.

        Args:
            image_path (str): The location of the image file.

        Returns:
            np.ndarray: The embedding vector.
        """
        return self.__predict(image_path)
    
    def create_search_kdtree(self, model_name):
        """
        Create a KDTree for searching by traversing the embeddings folder for the specified model.

        Args:
            model_name (str): The name of the model used to generate embeddings.

        Returns:
            KDTreeSearch: KDTreeSearch instance for querying embeddings.
        """
        points = []
        metadata_list = []

        # Traverse the embeddings directory for the specified model
        for embedding_path in glob(os.path.join(EMBEDDINGS_STORAGE, model_name, "*/*.npy")):
            # Load the embedding
            embedding = np.load(embedding_path)

            # Extract first name, last name, and image ID from the file path
            _, _, name, image_filename = embedding_path.split(os.path.sep)
            name_parts = name.split('_')
            first_name = name_parts[0]
            last_name = '_'.join(name_parts[1:])
            image_id = os.path.splitext(image_filename)[0]

            # Append the embedding and metadata to the lists
            points.append(embedding)
            metadata_list.append({"first_name": first_name, "last_name": last_name, "image_filename": image_id})

        # Construct the KD-Tree
        self.index = KDTree(k=256, points=points, metadata_list=metadata_list)
        self.search = KDTreeSearch(self.index, self.search.distance_func)
        print("KD-Tree constructed with embeddings from model:", model_name)
        return self.search
    
    def add_image(self, image_path):
        """
        Add a new image to the KDTree and update the index and search instance.

        Args:
            image_path (str): The file path to the new image.
        """
        # Extract first name, last name, and image ID from the file path
        _, subdir, name, image_filename = image_path.split(os.path.sep)
        name_parts = name.split('_')
        first_name = name_parts[0]
        last_name = '_'.join(name_parts[1:])
        image_id = os.path.splitext(image_filename)[0]

        # Preprocess the image and extract embeddings
        embedding = self.__predict(image_path)

        # Save the embeddings
        embedding_filename = os.path.join(EMBEDDINGS_STORAGE, self.model_name, f"{first_name}_{last_name}", f"{image_id}.npy")
        self.__save_embeddings(embedding_filename, embedding)

        # Append the embedding and metadata to the lists
        self.points.append(embedding)
        self.metadata.append({"first_name": first_name, "last_name": last_name, "image_filename": image_filename})

        # Update the KD-Tree
        self.index = KDTree(k=256, points=self.points, metadata_list=self.metadata)
        self.search = KDTreeSearch(self.index, self.search.distance_func)
        print(f"Added image {image_path} to KD-Tree.")

    def change_model(self, image_size, architecture):
        """
        Change the model architecture and image size.

        Args:
            image_size (int): The new image size.
            architecture (str): The new model architecture.
        """
        # Update preprocessing
        self.preprocessing = Preprocessing(image_size=image_size)

        # Update model
        model_name = f"model_size_{str(image_size).zfill(3)}_{architecture}"
        model_path = f"simclr_resources/{model_name}.pth"
        self.model = Model(model_path)
        self.model_name = model_name

        # Recompute the embeddings and rebuild the KD-Tree
        self.precompute()
        print(f"Changed model to {model_name} and re-computed embeddings.")

if __name__ == "__main__":
    os.chdir('visual_search_system/')
    image_size = 224
    architecture = 'resnet_034'
    model_name = f"model_size_{str(image_size).zfill(3)}_{architecture}"

    preprocessing = Preprocessing(image_size=image_size)
    model = Model(f"simclr_resources/{model_name}.pth")
    index = KDTree(k=256, points=[])
    search_euclidean = KDTreeSearch(index, Measure.euclidean)
    pipeline = Pipeline(preprocessing=preprocessing,
                        model=model,
                        index=index,
                        search=search_euclidean)

    # Precompute the embeddings and build the KD-Tree
    pipeline.precompute()

    # Example of searching the gallery
    probe_image_path = "simclr_resources/probes/Aaron_Sorkin/Aaron_Sorkin_0002.jpg"
    results = pipeline.search_gallery(probe_image_path)
    print("Nearest neighbors:")
    if results:
        for result in results:
            print(result)
    else:
        print("No neighbors found.")

    # Add a new image to the KDTree
    new_image_path = "storage/gallery/Mark_Warner/Mark_Warner_0001.jpg"
    pipeline.add_image(new_image_path)
