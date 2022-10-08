"""Structure to randomize images for discriminators"""
import random
import torch

class ImagePool():# pylint: disable=too-few-public-methods
    """This class implements an image buffer that stores previously generated
    images.

    This buffer enables us to update discriminators using a history of generated
    images rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameters
        ----------
        pool_size: int
            The size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        With probability 0.5, the buffer will return input images.
        With probability 0.5, the buffer will return images previously stored in
        the buffer, and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                # if the buffer is not full;
                #   keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                proba = random.uniform(0, 1)
                if proba > 0.5:
                    # the buffer will return a previously stored image,
                    # and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    #the buffer will return the current image
                    return_images.append(image)
        # collect all the images and return
        return_images = torch.cat(return_images, 0)
        return return_images
