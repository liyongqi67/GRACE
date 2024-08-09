import argparse
import json
import os
import uuid
import zipfile
from PIL import Image
import base64
from io import BytesIO

import braceexpand
import webdataset as wds
import pickle
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "--output_dir",
    type=str,
    help="Pass in the directory where the output shards (as tar files) will be written to.",
)
arg_parser.add_argument(
    "--json_file",
    type=str,
    help="image-caption pairs json_file",
)
arg_parser.add_argument(
    "--image_dir",
    type=str,
    help="Pass in the directory where the images have been downloaded to.",
)
arg_parser.add_argument(
    "--num_files_per_shard",
    type=int,
    default=5000,
)
args = arg_parser.parse_args()

def main():
    os.makedirs(args.output_dir, exist_ok=True)
    with wds.ShardWriter(args.output_dir + "/%09d.tar") as sink:
                with open(args.json_file, 'r') as f:
                    data = json.load(f)['images']
                idx = 0
                image_name2id_dict={}
                id2image_name_dict={}
                for original_sample_data in data:
                    if original_sample_data['filename'].split("_")[0] not in image_name2id_dict:
                        image_name2id_dict[original_sample_data['filename'].split("_")[0]] = len(image_name2id_dict)
                        id2image_name_dict[len(image_name2id_dict)-1] = original_sample_data['filename'].split("_")[0]
                with open(os.path.join(args.output_dir, "image_name2id_dict.pkl"), "wb") as tf:
                    print("len image_name2id_dict", len(image_name2id_dict))
                    pickle.dump(image_name2id_dict,tf)
                with open(os.path.join(args.output_dir, "id2image_name_dict.pkl"), "wb") as tf:
                    print("len id2image_name_dict", len(id2image_name_dict))
                    pickle.dump(id2image_name_dict,tf)

                image_name = "" ##########We set the same image for all samples as the pad image
                for original_sample_data in data:
                    if original_sample_data['split'] =='train':
                        if image_name =="":
                            image_name = original_sample_data['filename'].split("_")[0]
                        for sentence in original_sample_data['sentences']:
                            # get image names from json
                            sample_data = {'image_info':[{'face_detections': None, 'image_name': image_name, 'matched_sim': 0.2, 'matched_text_index': 0,'raw_url': 'none'}],
                                            'similarity_matrix': [[0.2]],
                                             'text_list':[sentence["raw"]+"#"+'id '+ str(image_name2id_dict[original_sample_data['filename'].split("_")[0]])],
                                             'url': 'none',
                                             'could_have_url_duplicate': 0}
                            image_info = sample_data["image_info"]
                            image_names = [image["image_name"] for image in image_info]
                            # Add each image to the tar file
                            for img_idx, image_name in enumerate(image_names):
                                try:
                                    # load image
                                    img = Image.open(
                                        os.path.join(args.image_dir, image_name)
                                    ).convert("RGB")
                                    buffered = BytesIO()
                                    img.save(buffered, format="JPEG")
                                    img_str = base64.b64encode(buffered.getvalue())

                                    # convert to base64
                                    sample_data["image_info"][img_idx][
                                        "image_base64"
                                    ] = img_str.decode("utf-8")
                                except FileNotFoundError:
                                    print(
                                        f"Did not find {image_name} downloaded. This can happen if the url is now 404."
                                    )
                                except Exception as e:
                                    print(f"Error processing {image_name}: {e}")
                            key_str = uuid.uuid4().hex
                            sink.write({"__key__": key_str, "json": sample_data})
                            idx+=1
                            if (idx + 1) % args.num_files_per_shard == 0:
                                sink.next_stream()
                with open("/storage_fast/yqli/project/AutoregressiveImageRetrieval/data/checkpoints/flicker30k_i2t_2/pseudo_query.json", 'r') as f:
                    data = json.load(f)
                    for line in data:
                        for sentence in line['caption']:
                            # get image names from json
                            sample_data = {'image_info':[{'face_detections': None, 'image_name': image_name, 'matched_sim': 0.2, 'matched_text_index': 0,'raw_url': 'none'}],
                                            'similarity_matrix': [[0.2]],
                                             'text_list':[sentence+"#"+'id '+ str(image_name2id_dict[line['image_id']+".jpg"])],
                                             'url': 'none',
                                             'could_have_url_duplicate': 0}
                            image_info = sample_data["image_info"]
                            image_names = [image["image_name"] for image in image_info]
                            # Add each image to the tar file
                            for img_idx, image_name in enumerate(image_names):
                                try:
                                    # load image
                                    img = Image.open(
                                        os.path.join(args.image_dir, image_name)
                                    ).convert("RGB")
                                    buffered = BytesIO()
                                    img.save(buffered, format="JPEG")
                                    img_str = base64.b64encode(buffered.getvalue())

                                    # convert to base64
                                    sample_data["image_info"][img_idx][
                                        "image_base64"
                                    ] = img_str.decode("utf-8")
                                except FileNotFoundError:
                                    print(
                                        f"Did not find {image_name} downloaded. This can happen if the url is now 404."
                                    )
                                except Exception as e:
                                    print(f"Error processing {image_name}: {e}")
                            key_str = uuid.uuid4().hex
                            sink.write({"__key__": key_str, "json": sample_data})
                            idx+=1
                            if (idx + 1) % args.num_files_per_shard == 0:
                                sink.next_stream()


# def main():
#     os.makedirs(args.output_dir, exist_ok=True)
#     with wds.ShardWriter(args.output_dir + "/%09d.tar") as sink:
#                 with open(args.json_file, 'r') as f:
#                     data = json.load(f)['images']
#                 idx = 0
#                 image_name2id_dict={}
#                 id2image_name_dict={}
#                 for original_sample_data in data:
#                     if original_sample_data['split'] == "train":
#                         if original_sample_data['filename'].split("_")[0] not in image_name2id_dict:
#                             image_name2id_dict[original_sample_data['filename'].split("_")[0]] = len(image_name2id_dict)
#                             id2image_name_dict[len(image_name2id_dict)-1] = original_sample_data['filename'].split("_")[0]
#                 for original_sample_data in data:
#                     if original_sample_data['split'] == "val":
#                         if original_sample_data['filename'].split("_")[0] not in image_name2id_dict:
#                             image_name2id_dict[original_sample_data['filename'].split("_")[0]] = len(image_name2id_dict)
#                             id2image_name_dict[len(image_name2id_dict)-1] = original_sample_data['filename'].split("_")[0]
#                 for original_sample_data in data:
#                     if original_sample_data['split'] == "test":
#                         if original_sample_data['filename'].split("_")[0] not in image_name2id_dict:
#                             image_name2id_dict[original_sample_data['filename'].split("_")[0]] = len(image_name2id_dict)
#                             id2image_name_dict[len(image_name2id_dict)-1] = original_sample_data['filename'].split("_")[0]
#                 with open(os.path.join(args.output_dir, "image_name2id_dict.pkl"), "wb") as tf:
#                     print("len image_name2id_dict", len(image_name2id_dict))
#                     pickle.dump(image_name2id_dict,tf)
#                 with open(os.path.join(args.output_dir, "id2image_name_dict.pkl"), "wb") as tf:
#                     print("len id2image_name_dict", len(id2image_name_dict))
#                     pickle.dump(id2image_name_dict,tf)
#                 for original_sample_data in data:
#                     if original_sample_data['split'] == "train":
#                         # get image names from json
#                         sample_data = {'image_info':[{'face_detections': None, 'image_name': original_sample_data['filename'].split("_")[0], 'matched_sim': 0.2, 'matched_text_index': 0,'raw_url': 'none'}],
#                                         'similarity_matrix': [[0.2]],
#                                          'text_list':['id '+ " ".join(list(str(image_name2id_dict[original_sample_data['filename'].split("_")[0]])))],
#                                          'url': 'none',
#                                          'could_have_url_duplicate': 0}
#                         image_info = sample_data["image_info"]
#                         image_names = [image["image_name"] for image in image_info]
#                         # Add each image to the tar file
#                         for img_idx, image_name in enumerate(image_names):
#                             try:
#                                 # load image
#                                 img = Image.open(
#                                     os.path.join(args.image_dir, image_name)
#                                 ).convert("RGB")
#                                 buffered = BytesIO()
#                                 img.save(buffered, format="JPEG")
#                                 img_str = base64.b64encode(buffered.getvalue())

#                                 # convert to base64
#                                 sample_data["image_info"][img_idx][
#                                     "image_base64"
#                                 ] = img_str.decode("utf-8")
#                             except FileNotFoundError:
#                                 print(
#                                     f"Did not find {image_name} downloaded. This can happen if the url is now 404."
#                                 )
#                             except Exception as e:
#                                 print(f"Error processing {image_name}: {e}")
#                         key_str = uuid.uuid4().hex
#                         sink.write({"__key__": key_str, "json": sample_data})
#                         idx+=1
#                         if (idx + 1) % args.num_files_per_shard == 0:
#                             sink.next_stream()
#                     if original_sample_data['split'] == "val":
#                         # get image names from json
#                         sample_data = {'image_info':[{'face_detections': None, 'image_name': original_sample_data['filename'].split("_")[0], 'matched_sim': 0.2, 'matched_text_index': 0,'raw_url': 'none'}],
#                                         'similarity_matrix': [[0.2]],
#                                          'text_list':['id '+ " ".join(list(str(image_name2id_dict[original_sample_data['filename'].split("_")[0]])))],
#                                          'url': 'none',
#                                          'could_have_url_duplicate': 0}
#                         image_info = sample_data["image_info"]
#                         image_names = [image["image_name"] for image in image_info]
#                         # Add each image to the tar file
#                         for img_idx, image_name in enumerate(image_names):
#                             try:
#                                 # load image
#                                 img = Image.open(
#                                     os.path.join(args.image_dir, image_name)
#                                 ).convert("RGB")
#                                 buffered = BytesIO()
#                                 img.save(buffered, format="JPEG")
#                                 img_str = base64.b64encode(buffered.getvalue())

#                                 # convert to base64
#                                 sample_data["image_info"][img_idx][
#                                     "image_base64"
#                                 ] = img_str.decode("utf-8")
#                             except FileNotFoundError:
#                                 print(
#                                     f"Did not find {image_name} downloaded. This can happen if the url is now 404."
#                                 )
#                             except Exception as e:
#                                 print(f"Error processing {image_name}: {e}")
#                         key_str = uuid.uuid4().hex
#                         sink.write({"__key__": key_str, "json": sample_data})
#                         idx+=1
#                         if (idx + 1) % args.num_files_per_shard == 0:
#                             sink.next_stream()

#                     if original_sample_data['split'] == "test":
#                         # get image names from json   " ".join(list(str(image_name2id_dict[original_sample_data['filename'].split("_")[0]])))
#                         sample_data = {'image_info':[{'face_detections': None, 'image_name': original_sample_data['filename'].split("_")[0], 'matched_sim': 0.2, 'matched_text_index': 0,'raw_url': 'none'}],
#                                         'similarity_matrix': [[0.2]],
#                                          'text_list':['id '+ " ".join(list(str(image_name2id_dict[original_sample_data['filename'].split("_")[0]])))],
#                                          'url': 'none',
#                                          'could_have_url_duplicate': 0}
#                         image_info = sample_data["image_info"]
#                         image_names = [image["image_name"] for image in image_info]
#                         # Add each image to the tar file
#                         for img_idx, image_name in enumerate(image_names):
#                             try:
#                                 # load image
#                                 img = Image.open(
#                                     os.path.join(args.image_dir, image_name)
#                                 ).convert("RGB")
#                                 buffered = BytesIO()
#                                 img.save(buffered, format="JPEG")
#                                 img_str = base64.b64encode(buffered.getvalue())

#                                 # convert to base64
#                                 sample_data["image_info"][img_idx][
#                                     "image_base64"
#                                 ] = img_str.decode("utf-8")
#                             except FileNotFoundError:
#                                 print(
#                                     f"Did not find {image_name} downloaded. This can happen if the url is now 404."
#                                 )
#                             except Exception as e:
#                                 print(f"Error processing {image_name}: {e}")
#                         key_str = uuid.uuid4().hex
#                         sink.write({"__key__": key_str, "json": sample_data})
#                         idx+=1
#                         if (idx + 1) % args.num_files_per_shard == 0:
#                             sink.next_stream()

if __name__ == "__main__":
    main()