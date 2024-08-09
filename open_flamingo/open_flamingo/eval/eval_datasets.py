import json
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from open_flamingo.eval.classification_utils import IMAGENET_1K_CLASS_ID_TO_LABEL

import pickle



class I2idDataset(Dataset):
    def __init__(
        self,
        image_train_dir_path,
        annotations_path,
        is_train,
        dataset_name,
        image_val_dir_path=None,
        image_name2id_dict=None
    ):
        self.image_train_dir_path = image_train_dir_path
        self.image_val_dir_path = image_val_dir_path
        self.annotations = []
        self.is_train = is_train
        self.dataset_name = dataset_name

        full_annotations = json.load(open(annotations_path))["images"]

        for i in range(len(full_annotations)):
            if self.is_train and full_annotations[i]["split"] != "train":
                continue
            elif not self.is_train and full_annotations[i]["split"] != "test":
                continue

            self.annotations.append(full_annotations[i])
        with open(image_name2id_dict, 'rb') as f:
            self.image_name2id_dict = pickle.load(f)
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if self.dataset_name == "coco":
            image = Image.open(
                os.path.join(
                    self.image_train_dir_path, self.annotations[idx]["filename"]
                )
                if self.annotations[idx]["filepath"] == "train2014"
                else os.path.join(
                    self.image_val_dir_path, self.annotations[idx]["filename"]
                )
            )
        elif self.dataset_name == "flickr":
            image = Image.open(
                os.path.join(
                    self.image_train_dir_path, self.annotations[idx]["filename"]
                )
            )
        image.load()
        caption = self.annotations[idx]["sentences"][0]["raw"]
        return {
            "image": image,
            "id":  self.image_name2id_dict[self.annotations[idx]["filename"]],               ############identifier id
            "image_id": self.annotations[idx]["cocoid"]
            if self.dataset_name == "coco"
            else self.annotations[idx]["filename"].split(".")[0],
        }


class T2idDataset(Dataset):
    def __init__(
        self,
        image_train_dir_path,
        annotations_path,
        is_train,
        dataset_name,
        image_val_dir_path=None,
        image_name2id_dict=None
    ):
        self.image_train_dir_path = image_train_dir_path
        self.image_val_dir_path = image_val_dir_path
        self.annotations = []
        self.is_train = is_train
        self.dataset_name = dataset_name

        full_annotations = json.load(open(annotations_path))["images"]
        image_name = "" ##########We set the same image for all samples as the pad image
        for original_sample_data in full_annotations:
            if original_sample_data['split'] =='train':
                if image_name =="":
                    image_name = original_sample_data['filename']
                    break
        self.image_name = image_name
        for i in range(len(full_annotations)):
            if self.is_train and full_annotations[i]["split"] != "train":
                continue
            elif not self.is_train and full_annotations[i]["split"] != "test":
                continue
            for caption in full_annotations[i]["sentences"]:
                if self.dataset_name == "coco":
                    new_data={"filepath":full_annotations[i]["filepath"],"filename":full_annotations[i]["filename"],"caption":caption["raw"], "cocoid":full_annotations[i]["cocoid"]}
                else:
                    new_data={"filename":full_annotations[i]["filename"],"caption":caption["raw"]}
                self.annotations.append(new_data)
        if is_train:
            self.annotations = self.annotations[:1000]
        with open(image_name2id_dict, 'rb') as f:
            self.image_name2id_dict = pickle.load(f)
            
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if self.dataset_name == "coco":
            image = Image.open(
                os.path.join(
                    self.image_train_dir_path, self.image_name
                )
            )
        elif self.dataset_name == "flickr":
            image = Image.open(
                os.path.join(
                    self.image_train_dir_path, self.image_name
                )
            )


        image.load()
        return {
            "image": image,
            "id":  self.image_name2id_dict[self.annotations[idx]["filename"]],               ############identifier id
            "caption":  self.annotations[idx]["caption"],
            "image_id": self.annotations[idx]["cocoid"]
            if self.dataset_name == "coco"
            else self.annotations[idx]["filename"].split(".")[0],
            "filename": self.annotations[idx]["filename"]
        }


class I2tDataset(Dataset):
    def __init__(
        self,
        image_train_dir_path,
        annotations_path,
        is_train,
        dataset_name,
        image_val_dir_path=None,
    ):
        self.image_train_dir_path = image_train_dir_path
        self.image_val_dir_path = image_val_dir_path
        self.annotations = []
        self.is_train = is_train
        self.dataset_name = dataset_name

        full_annotations = json.load(open(annotations_path))["images"]

        for i in range(len(full_annotations)):
            if self.is_train and full_annotations[i]["split"] != "train":
                continue
            elif not self.is_train and full_annotations[i]["split"] != "test":
                continue

            self.annotations.append(full_annotations[i])
        if is_train:
            self.annotations = self.annotations[:1000]
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if self.dataset_name == "coco":
            image = Image.open(
                os.path.join(
                    self.image_train_dir_path, self.annotations[idx]["filename"]
                )
                if self.annotations[idx]["filepath"] == "train2014"
                else os.path.join(
                    self.image_val_dir_path, self.annotations[idx]["filename"]
                )
            )
        elif self.dataset_name == "flickr":
            image = Image.open(
                os.path.join(
                    self.image_train_dir_path, self.annotations[idx]["filename"]
                )
            )
        image.load()
        caption_list = []
        for c in self.annotations[idx]["sentences"]:
            caption_list.append({'raw':c['raw']})
        return {
            "image": image,
            "caption":  caption_list,
            "image_id": self.annotations[idx]["cocoid"]
            if self.dataset_name == "coco"
            else self.annotations[idx]["filename"].split(".")[0],
        }
class Id2CaptionDataset(Dataset):
    def __init__(
        self,
        image_train_dir_path,
        annotations_path,
        is_train,
        dataset_name,
        image_val_dir_path=None,
        image_name2id_dict=None
    ):
        self.image_train_dir_path = image_train_dir_path
        self.image_val_dir_path = image_val_dir_path
        self.annotations = []
        self.is_train = is_train
        self.dataset_name = dataset_name

        full_annotations = json.load(open(annotations_path))["images"]
        image_name = "" ##########We set the same image for all samples as the pad image
        for original_sample_data in full_annotations:
            if original_sample_data['split'] =='train':
                if image_name =="":
                    image_name = original_sample_data['filename']
                    break
        self.image_name = image_name
        for i in range(len(full_annotations)):
            if self.is_train and full_annotations[i]["split"] != "train":
                continue
            elif not self.is_train and full_annotations[i]["split"] != "test":
                continue
            self.annotations.append(full_annotations[i])
        if is_train:
            self.annotations = self.annotations[:1000]
        with open(image_name2id_dict, 'rb') as f:
            self.image_name2id_dict = pickle.load(f)
            
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if self.dataset_name == "coco":
            image = Image.open(
                os.path.join(
                    self.image_train_dir_path, self.image_name
                )
            )
        elif self.dataset_name == "flickr":
            image = Image.open(
                os.path.join(
                    self.image_train_dir_path, self.image_name
                )
            )


        image.load()
        caption_list = []
        for c in self.annotations[idx]["sentences"]:
            caption_list.append({'raw':c['raw']})
        return {
            "image": image,
            "id":  self.image_name2id_dict[self.annotations[idx]["filename"]],               ############identifier id
            "caption":  caption_list,
            "image_id": self.annotations[idx]["cocoid"]
            if self.dataset_name == "coco"
            else self.annotations[idx]["filename"].split(".")[0],
            "filename": self.annotations[idx]["filename"]
        }
class CaptionDataset(Dataset):
    def __init__(
        self,
        image_train_dir_path,
        annotations_path,
        is_train,
        dataset_name,
        image_val_dir_path=None,
    ):
        self.image_train_dir_path = image_train_dir_path
        self.image_val_dir_path = image_val_dir_path
        self.annotations = []
        self.is_train = is_train
        self.dataset_name = dataset_name

        full_annotations = json.load(open(annotations_path))["images"]

        for i in range(len(full_annotations)):
            if self.is_train and full_annotations[i]["split"] != "train":
                continue
            elif not self.is_train and full_annotations[i]["split"] != "test":
                continue

            self.annotations.append(full_annotations[i])
        if is_train:
            self.annotations = self.annotations[:1000]
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if self.dataset_name == "coco":
            image = Image.open(
                os.path.join(
                    self.image_train_dir_path, self.annotations[idx]["filename"]
                )
                if self.annotations[idx]["filepath"] == "train2014"
                else os.path.join(
                    self.image_val_dir_path, self.annotations[idx]["filename"]
                )
            )
        elif self.dataset_name == "flickr":
            image = Image.open(
                os.path.join(
                    self.image_train_dir_path, self.annotations[idx]["filename"]
                )
            )
        image.load()
        caption_list = []
        for c in self.annotations[idx]["sentences"]:
            caption_list.append({'raw':c['raw']})
        return {
            "image": image,
            "caption": caption_list,
            "image_id": self.annotations[idx]["cocoid"]
            if self.dataset_name == "coco"
            else self.annotations[idx]["filename"],
        }


class VQADataset(Dataset):
    def __init__(
        self, image_dir_path, question_path, annotations_path, is_train, dataset_name
    ):
        self.questions = json.load(open(question_path, "r"))["questions"]
        if annotations_path is not None:
            self.answers = json.load(open(annotations_path, "r"))["annotations"]
        else:
            self.answers = None
        self.image_dir_path = image_dir_path
        self.is_train = is_train
        self.dataset_name = dataset_name
        if self.dataset_name in {"vqav2", "ok_vqa"}:
            self.img_coco_split = self.image_dir_path.strip("/").split("/")[-1]
            assert self.img_coco_split in {"train2014", "val2014", "test2015"}

    def __len__(self):
        return len(self.questions)

    def get_img_path(self, question):
        if self.dataset_name in {"vqav2", "ok_vqa"}:
            return os.path.join(
                self.image_dir_path,
                f"COCO_{self.img_coco_split}_{question['image_id']:012d}.jpg"
                if self.is_train
                else f"COCO_{self.img_coco_split}_{question['image_id']:012d}.jpg",
            )
        elif self.dataset_name == "vizwiz":
            return os.path.join(self.image_dir_path, question["image_id"])
        elif self.dataset_name == "textvqa":
            return os.path.join(self.image_dir_path, f"{question['image_id']}.jpg")
        else:
            raise Exception(f"Unknown VQA dataset {self.dataset_name}")

    def __getitem__(self, idx):
        question = self.questions[idx]
        img_path = self.get_img_path(question)
        image = Image.open(img_path)
        image.load()
        results = {
            "image": image,
            "question": question["question"],
            "question_id": question["question_id"],
        }
        if self.answers is not None:
            answers = self.answers[idx]
            results["answers"] = [a["answer"] for a in answers["answers"]]
        return results


class ImageNetDataset(ImageFolder):
    """Class to represent the ImageNet1k dataset."""

    def __init__(self, root, **kwargs):
        super().__init__(root=root, **kwargs)

    def __getitem__(self, idx):
        sample, target = super().__getitem__(idx)
        target_label = IMAGENET_1K_CLASS_ID_TO_LABEL[target]
        return {
            "id": idx,
            "image": sample,
            "class_id": target,  # numeric ID of the ImageNet class
            "class_name": target_label,  # human-readable name of ImageNet class
        }


class HatefulMemesDataset(Dataset):
    def __init__(self, image_dir_path, annotations_path):
        self.image_dir_path = image_dir_path
        with open(annotations_path, "r") as f:
            self.annotations = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.image_dir_path, annotation["img"].split("/")[-1])
        image = Image.open(img_path)
        image.load()
        return {
            "id": annotation["id"],
            "image": image,
            "ocr": annotation["text"],
            "class_name": "yes" if annotation["label"] == 1 else "no",
            "class_id": annotation["label"],
        }
