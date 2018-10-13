import torchvision.datasets as datasets
import random

from collections import defaultdict

class TripletImageNetDataset(datasets.ImageFolder):

    def __init__(self, root, n_triplets_per_sample, transform=None):
        super(TripletImageNetDataset, self).__init__(root, transform)
        self.n_triplets_per_sample = n_triplets_per_sample
        self.training_triplets = self.generate_triplets(self.imgs, self.n_triplets_per_sample, len(self.classes))

    @staticmethod
    def generate_triplets(imgs, n_triplets_per_sample, n_classes):
        def create_indices(_imgs):
            indices = defaultdict(list)
            for img_path, label in _imgs:
                indices[label].append(img_path)
            return indices

        triplets = []
        indices = create_indices(imgs)

        for pos_class, imgs in indices.items():
            for anchor_img in imgs:
                for _ in range(n_triplets_per_sample):
                    pos_img = random.choice(imgs)
                    while anchor_img == pos_img:
                        pos_img = random.choice(imgs)

                    neg_class = random.randint(0, n_classes-1)
                    while pos_class == neg_class:
                        neg_class = random.randint(0, n_classes-1)

                    neg_img = random.choice(indices[neg_class])
    
                    triplets.append([anchor_img, pos_img, neg_img, pos_class, neg_class])
        return triplets

    def __getitem__(self, index):
        def transform(img_path):
            img = self.loader(img_path)
            return self.transform(img)

        a, p, n, c1, c2 = self.training_triplets[index]

        img_a, img_p, img_n = transform(a), transform(p), transform(n)
        return img_a, img_p, img_n, c1, c2, a


    def __len__(self):
        return len(self.training_triplets)

    def get_idx_to_class(self):
        idx_to_class = dict()
        for key, val in self.class_to_idx.itmes():
            idx_to_class[val] = key
        return idx_to_class
    
    def get_class_to_idx(self):
        return self.class_to_idx