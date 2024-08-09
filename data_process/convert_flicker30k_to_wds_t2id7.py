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
from random import choice
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
arg_parser.add_argument(
    "--identifier_type",
    choices=["semantic_identifier","automatic_identifier", "structured_identifier_for_caption", "structured_identifier", "string_identifier", "numeric_identifier"]
)
arg_parser.add_argument(
    "--image_name2id_dict",
    type=str,
)
arg_parser.add_argument(
    "--pseudo_query",
    type=str,
)
arg_parser.add_argument(
    "--coco",
    action="store_true",
)
args = arg_parser.parse_args()
with open(args.image_name2id_dict, 'rb') as f:
    image_name2id_dict = pickle.load(f)
def main():
    if args.identifier_type == "structured_identifier":
        os.makedirs(args.output_dir, exist_ok=True)
        with wds.ShardWriter(args.output_dir + "/%09d.tar") as sink:
                    with open(args.json_file, 'r') as f:
                        data = json.load(f)['images']
                    idx = 0
                    image_name = "" ##########We set the same image for all samples as the pad image
                    for original_sample_data in data:
                        if original_sample_data['split'] =='train':
                            if image_name =="":
                                image_name = original_sample_data['filename']
                            for sentence in original_sample_data['sentences']:
                                # get image names from json
                                sample_data = {'image_info':[{'face_detections': None, 'image_name': image_name, 'matched_sim': 0.2, 'matched_text_index': 0,'raw_url': 'none'}],
                                                'similarity_matrix': [[0.2]],
                                                 'text_list':[sentence["raw"]+"#"+'id '+ image_name2id_dict[original_sample_data['filename']]],
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
                    with open(args.pseudo_query, 'r') as f:
                        data = json.load(f)
                        for line in data:
                            for sentence in line['caption']:
                                # get image names from json
                                if args.coco:
                                    sample_data = {'image_info':[{'face_detections': None, 'image_name': image_name, 'matched_sim': 0.2, 'matched_text_index': 0,'raw_url': 'none'}],
                                                    'similarity_matrix': [[0.2]],
                                                     'text_list':[sentence+"#"+'id '+ image_name2id_dict["COCO_val2014_" + str(line['image_id']).zfill(12)+".jpg"]],
                                                     'url': 'none',
                                                     'could_have_url_duplicate': 0}
                                else:
                                    sample_data = {'image_info':[{'face_detections': None, 'image_name': image_name, 'matched_sim': 0.2, 'matched_text_index': 0,'raw_url': 'none'}],
                                                    'similarity_matrix': [[0.2]],
                                                     'text_list':[sentence+"#"+'id '+ image_name2id_dict[str(line['image_id'])+".jpg"]],
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
    if args.identifier_type == "semantic_identifier":
        os.makedirs(args.output_dir, exist_ok=True)
        with wds.ShardWriter(args.output_dir + "/%09d.tar") as sink:
                    with open(args.json_file, 'r') as f:
                        data = json.load(f)['images']
                    idx = 0
                    image_name = "" ##########We set the same image for all samples as the pad image
                    for original_sample_data in data:
                        if original_sample_data['split'] =='train':
                            if image_name =="":
                                image_name = original_sample_data['filename']
                            for sentence in original_sample_data['sentences']:
                                # get image names from json
                                sample_data = {'image_info':[{'face_detections': None, 'image_name': image_name, 'matched_sim': 0.2, 'matched_text_index': 0,'raw_url': 'none'}],
                                                'similarity_matrix': [[0.2]],
                                                 'text_list':[sentence["raw"]+"#"+'id '+ choice(image_name2id_dict[original_sample_data['filename']])],
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
    if args.identifier_type == "structured_identifier_for_caption":
        os.makedirs(args.output_dir, exist_ok=True)
        with wds.ShardWriter(args.output_dir + "/%09d.tar") as sink:
                    with open(args.json_file, 'r') as f:
                        data = json.load(f)['images']
                    idx = 0

                    image_name = "" ##########We set the same image for all samples as the pad image
                    for original_sample_data in data:
                        if original_sample_data['split'] =='train':
                            if image_name =="":
                                image_name = original_sample_data['filename']
                            for sentence in original_sample_data['sentences']:
                                # get image names from json
                                sample_data = {'image_info':[{'face_detections': None, 'image_name': image_name, 'matched_sim': 0.2, 'matched_text_index': 0,'raw_url': 'none'}],
                                                'similarity_matrix': [[0.2]],
                                                 'text_list':['Describe the image id '+ image_name2id_dict[original_sample_data['filename']] +"#" + sentence["raw"]],
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
    if args.identifier_type == "automatic_identifier":
        os.makedirs(args.output_dir, exist_ok=True)
        with wds.ShardWriter(args.output_dir + "/%09d.tar") as sink:
                    with open(args.json_file, 'r') as f:
                        data = json.load(f)['images']
                    idx = 0
                    image_name = "" ##########We set the same image for all samples as the pad image
                    for original_sample_data in data:
                        if original_sample_data['split'] =='train':
                            if image_name =="":
                                image_name = original_sample_data['filename']
                            for sentence in original_sample_data['sentences']:
                                # get image names from json
                                sample_data = {'image_info':[{'face_detections': None, 'image_name': image_name, 'matched_sim': 0.2, 'matched_text_index': 0,'raw_url': 'none'}],
                                                'similarity_matrix': [[0.2]],
                                                 'text_list':[sentence["raw"]+"#"+'id '+ image_name2id_dict[original_sample_data['filename']]],
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
                    with open(args.pseudo_query, 'r') as f:
                        data = json.load(f)
                        for line in data:
                            for sentence in line['caption']:
                                # get image names from json
                                if args.coco:
                                    sample_data = {'image_info':[{'face_detections': None, 'image_name': image_name, 'matched_sim': 0.2, 'matched_text_index': 0,'raw_url': 'none'}],
                                                    'similarity_matrix': [[0.2]],
                                                     'text_list':[sentence+"#"+'id '+ image_name2id_dict["COCO_val2014_" + str(line['image_id']).zfill(12)+".jpg"]],
                                                     'url': 'none',
                                                     'could_have_url_duplicate': 0}
                                else:
                                    sample_data = {'image_info':[{'face_detections': None, 'image_name': image_name, 'matched_sim': 0.2, 'matched_text_index': 0,'raw_url': 'none'}],
                                                    'similarity_matrix': [[0.2]],
                                                     'text_list':[sentence+"#"+'id '+ image_name2id_dict[str(line['image_id'])+".jpg"]],
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
    if args.identifier_type == "string_identifier":
        os.makedirs(args.output_dir, exist_ok=True)
        with wds.ShardWriter(args.output_dir + "/%09d.tar") as sink:
                    with open(args.json_file, 'r') as f:
                        data = json.load(f)['images']
                    idx = 0
                    image_name = "" ##########We set the same image for all samples as the pad image
                    for original_sample_data in data:
                        if original_sample_data['split'] =='train':
                            if image_name =="":
                                image_name = original_sample_data['filename']
                            for sentence in original_sample_data['sentences']:
                                # get image names from json
                                sample_data = {'image_info':[{'face_detections': None, 'image_name': image_name, 'matched_sim': 0.2, 'matched_text_index': 0,'raw_url': 'none'}],
                                                'similarity_matrix': [[0.2]],
                                                 'text_list':[sentence["raw"]+"#"+'id '+ image_name2id_dict[original_sample_data['filename']]],
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
                    with open(args.pseudo_query, 'r') as f:
                        data = json.load(f)
                        for line in data:
                            for sentence in line['caption']:
                                # get image names from json
                                if args.coco:
                                    sample_data = {'image_info':[{'face_detections': None, 'image_name': image_name, 'matched_sim': 0.2, 'matched_text_index': 0,'raw_url': 'none'}],
                                                    'similarity_matrix': [[0.2]],
                                                     'text_list':[sentence+"#"+'id '+ image_name2id_dict["COCO_val2014_" + str(line['image_id']).zfill(12)+".jpg"]],
                                                     'url': 'none',
                                                     'could_have_url_duplicate': 0}
                                else:
                                    sample_data = {'image_info':[{'face_detections': None, 'image_name': image_name, 'matched_sim': 0.2, 'matched_text_index': 0,'raw_url': 'none'}],
                                                    'similarity_matrix': [[0.2]],
                                                     'text_list':[sentence+"#"+'id '+ image_name2id_dict[str(line['image_id'])+".jpg"]],
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
    if args.identifier_type == "numeric_identifier":
        os.makedirs(args.output_dir, exist_ok=True)
        with wds.ShardWriter(args.output_dir + "/%09d.tar") as sink:
                    with open(args.json_file, 'r') as f:
                        data = json.load(f)['images']
                    idx = 0
                    image_name = "" ##########We set the same image for all samples as the pad image
                    for original_sample_data in data:
                        if original_sample_data['split'] =='train':
                            if image_name =="":
                                image_name = original_sample_data['filename']
                            for sentence in original_sample_data['sentences']:
                                # get image names from json
                                sample_data = {'image_info':[{'face_detections': None, 'image_name': image_name, 'matched_sim': 0.2, 'matched_text_index': 0,'raw_url': 'none'}],
                                                'similarity_matrix': [[0.2]],
                                                 'text_list':[sentence["raw"]+"#"+'id '+ image_name2id_dict[original_sample_data['filename']]],
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
                    with open(args.pseudo_query, 'r') as f:
                        data = json.load(f)
                        for line in data:
                            for sentence in line['caption']:
                                # get image names from json
                                if args.coco:
                                    sample_data = {'image_info':[{'face_detections': None, 'image_name': image_name, 'matched_sim': 0.2, 'matched_text_index': 0,'raw_url': 'none'}],
                                                    'similarity_matrix': [[0.2]],
                                                     'text_list':[sentence+"#"+'id '+ image_name2id_dict["COCO_val2014_" + str(line['image_id']).zfill(12)+".jpg"]],
                                                     'url': 'none',
                                                     'could_have_url_duplicate': 0}
                                else:
                                    sample_data = {'image_info':[{'face_detections': None, 'image_name': image_name, 'matched_sim': 0.2, 'matched_text_index': 0,'raw_url': 'none'}],
                                                    'similarity_matrix': [[0.2]],
                                                     'text_list':[sentence+"#"+'id '+ image_name2id_dict[str(line['image_id'])+".jpg"]],
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
if __name__ == "__main__":
    main()