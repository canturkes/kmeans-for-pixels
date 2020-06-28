import numpy as np
from PIL import Image
import sys

class KMeansForPixels(): ## can be used for finding K dominant colors in an image

    def __init__(self, image:Image.Image, K:int, max_iter=1000, resize_to=(100,100)):

        '''
        image: PIL.Image.Image object
        K: int, number of clusters
        max_iter: int, maximum number of iterations allowed
        resize_to: tuple of ints, initial resizing of the image

        Please note that the results are strongly dependent
        on centroid initialization.

        ''' 

        if K <= 0:
            print('K cannot be nonnegative!')
            raise ValueError
        else:
            self.orig_im = image
            self.mini_im = image.resize(resize_to)
            self.__im_data = np.asarray(self.mini_im).reshape(-1, 3)
            self.centroids, self.memberships, self.sd_error = self.__k_colors(self.__im_data, K, max_iter)

    ## k-means implementation
    def __SDE(self, data, members, centers, K):

        _sde = 0
        for i in range(K): 
            _sde += np.sum(np.linalg.norm(data[members==i]-centers[i], axis=1)) 
        return _sde

    def __update_memberships(self, data, centroids, n):
        new_memberships = np.empty(shape=n, dtype=np.uint16) ## there can be upmost 65536 diff. colors
        for i, point in enumerate(data): 
            new_memberships[i] = np.linalg.norm(centroids-point, axis=1).argmin()
            ## axis = 1 is crucial, don't forget  
        return new_memberships

    def __update_centroids(self, data, memberships, K):
        new_centroids = np.empty(shape=(K,3))
        for i in range(K):
            new_centroids[i] = np.sum(data[memberships==i], axis=0) / np.sum(memberships==i)
        return new_centroids

    def __init_centers(self, data, K, n):
        rng = np.random.default_rng()
        random_indices = rng.choice(n,K, replace=False)        
        return data[random_indices]

    def __k_colors(self, data, K, max_iter):
        num_of_px = len(data) ## prevent calling len() multiple times
        members = np.empty(num_of_px)
        sd_error = .0
        print('Clustering starts...')
        while True:
            with np.errstate(invalid='raise'):
                try:
                    centroids = self.__init_centers(data, K, num_of_px) 
                    for _ in range(max_iter): # _ stands for iteration number
                        members = self.__update_memberships(data, centroids, num_of_px)
                        centroids = self.__update_centroids (data, members, K)
                        new_error = self.__SDE(data, members, centroids, K)
                        if new_error != sd_error:
                            sd_error = new_error
                        else:
                            print('Converged!')
                            break
                    print('Done!')
                    break
                except FloatingPointError:
                    print('Orphan cluster error, starting over...')
        return np.floor(centroids).astype(np.uint8), members, sd_error
    ## end k-means implementation

## end class

## runtime example : >python KmeansForPixels.py <filename> <K>
if __name__ == "__main__":
    try:
        imfile = sys.argv[1]
        K = int(sys.argv[2])
        image = Image.open(imfile)
        dominant_colors = KMeansForPixels(image, K).centroids
        print(dominant_colors)
    except IndexError:
        print('Not enough args.')
        print('Quitting...')
    except IOError:
        print('File not found.')
        print('Quitting...')