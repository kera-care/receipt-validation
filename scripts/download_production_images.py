import os
import json
from PIL import Image
from google.cloud import storage

from google.api_core.exceptions import NotFound
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import List, Dict, Tuple


def is_valid_image_file(file_path: str) -> bool:
    
    try:
        image = Image.open(file_path).convert("RGB")
        image.verify()  # Verify that it is, in fact an image
        return True
    except (IOError, SyntaxError) as e:
        print(f"Invalid image file {file_path}: {e}")
        return False


def _download_single_image(client: storage.Client, bucket_name: str, output_dir: str, image_url: str) -> Tuple[bool, str, str]:
    """
    Download a single image from GCS bucket.
    
    Args:
        client: GCS client
        bucket_name: Name of the GCS bucket
        output_dir: Local directory to save images
        image_url: URL of the image to download
    
    Returns:
        Tuple[bool, str, str]: (success, image_url, error_message)
    """
    try:
        bucket = client.bucket(bucket_name)
        
        if image_url.startswith("gs://"):
            image_relative_url = image_url.replace("gs://", "").split("/", 1)[1]
        else:
            # Assume the image url is already relative url
            image_relative_url = image_url
            
        blob = bucket.blob(image_relative_url)
        output_path = os.path.join(output_dir, image_relative_url)
        output_path_dir = os.path.dirname(output_path)
        
        # Thread-safe directory creation
        if not os.path.exists(output_path_dir):
            os.makedirs(output_path_dir, exist_ok=True)
        
        # Check if file already exists and is valid
        if os.path.exists(output_path):
            if is_valid_image_file(output_path):
                return True, image_url, "Already exists"
            # Remove corrupted file
            print(f"Removing corrupted file: {output_path}")
            os.remove(output_path)
        
        # Download the image
        blob.download_to_filename(output_path)
        
        # Verify the downloaded image
        if is_valid_image_file(output_path):
            return True, image_url, "Downloaded successfully"
        else:
            os.remove(output_path)  # Remove invalid file
            return False, image_url, "Downloaded file is corrupted"
            
    except (storage.exceptions.InvalidResponse, NotFound) as e:
        return False, image_url, f"GCS error: {str(e)}"
    except Exception as e:
        return False, image_url, f"Unexpected error: {str(e)}"

def _download_task_images_threaded(client: storage.Client, bucket_name: str, output_dir: str, 
                                 task: dict, max_workers: int = 5, image_urls_key: str = "prescription_image_urls"
                                 ) -> Tuple[int, int, List[str]]:
    """
    Download all images for a single task using multithreading.
    
    Args:
        client: GCS client
        bucket_name: Name of the GCS bucket
        output_dir: Local directory to save images
        task: Task dictionary containing prescription_image_urls
        max_workers: Maximum number of concurrent downloads per task
        image_urls_key: Key in task dict containing image URLs
    
    Returns:
        Tuple[int, int, List[str]]: (successful_downloads, total_images, failed_urls)
    """
    image_urls = task.get(image_urls_key, [])
    task_id = task.get("task_id", "unknown")
    
    if not image_urls:
        return 0, 0, []
    
    successful = 0
    failed_urls = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download jobs for this task
        future_to_url = {
            executor.submit(_download_single_image, client, bucket_name, output_dir, url): url 
            for url in image_urls
        }
        
        # Collect results
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                success, image_url, message = future.result()
                if success:
                    successful += 1
                else:
                    failed_urls.append(f"{url}: {message}")
            except Exception as e:
                failed_urls.append(f"{url}: Exception in thread: {str(e)}")
    
    return successful, len(image_urls), failed_urls


def download_tasks_images(client: storage.Client, bucket_name: str, output_dir: str, 
                         tasks: List[Dict], max_workers: int = 20, images_per_task_workers: int = 5,
                         image_urls_key: str = "prescription_image_urls"
                        ) -> Dict[str, any]:
    """
    Download images for multiple tasks using multithreading.
    
    Args:
        client: GCS client
        bucket_name: Name of the GCS bucket  
        output_dir: Local directory to save images
        tasks: List of task dictionaries
        max_workers: Maximum number of concurrent tasks to process
        images_per_task_workers: Maximum number of concurrent downloads per task
        image_urls_key: Key in task dict containing image URLs
        
    Returns:
        Dict with download statistics
    """
    print(f"Downloading images for {len(tasks)} tasks using {max_workers} workers...")
    
    total_successful = 0
    total_images = 0
    total_failed = 0
    all_failed_urls = []
    
    # Create a thread-safe lock for updating progress
    progress_lock = threading.Lock()
    
    def process_task(task):
        successful, task_images, failed_urls = _download_task_images_threaded(
            client, bucket_name, output_dir, task, images_per_task_workers, image_urls_key=image_urls_key
        )
        return successful, task_images, len(failed_urls), failed_urls
    
    # Use ThreadPoolExecutor for task-level parallelism
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(process_task, task): task 
            for task in tasks
        }
        
        # Progress tracking
        with tqdm(total=len(tasks), desc="Processing tasks", unit="task") as pbar:
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                task_id = task.get("task_id", "unknown")
                
                try:
                    successful, task_images, failed_count, failed_urls = future.result()
                    
                    with progress_lock:
                        total_successful += successful
                        total_images += task_images
                        total_failed += failed_count
                        all_failed_urls.extend(failed_urls)
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'Success': f"{total_successful}/{total_images}",
                        'Failed': total_failed
                    })
                    
                except Exception as e:
                    print(f"Error processing task {task_id}: {str(e)}")
                    with progress_lock:
                        total_failed += len(task.get(image_urls_key, []))
                
                pbar.update(1)
    
    # Print summary
    print(f"\n{'='*80}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*80}")
    print(f"Total images: {total_images}")
    print(f"Successfully downloaded: {total_successful}")
    print(f"Failed downloads: {total_failed}")
    print(f"Success rate: {100 * total_successful / max(1, total_images):.1f}%")
    
    if all_failed_urls:
        print(f"\nFailed URLs (showing first 10):")
        for url in all_failed_urls[:10]:
            print(f"  - {url}")
        if len(all_failed_urls) > 10:
            print(f"  ... and {len(all_failed_urls) - 10} more")
    
    print(f"{'='*80}\n")
    
    return {
        'total_images': total_images,
        'successful': total_successful,
        'failed': total_failed,
        'success_rate': total_successful / max(1, total_images),
        'failed_urls': all_failed_urls
    }


def download_tasks_images_fast(client: storage.Client, bucket_name: str, output_dir: str, tasks: List[Dict], image_urls_key: str = "prescription_image_urls") -> Dict[str, any]:
    """
    Fast download with aggressive threading (use with caution on rate-limited APIs).
    
    Args:
        client: GCS client
        bucket_name: Name of the GCS bucket
        output_dir: Local directory to save images  
        tasks: List of task dictionaries
        image_urls_key: Key in task dict containing image URLs
        
    Returns:
        Dict with download statistics
    """
    return download_tasks_images(client, bucket_name, output_dir, tasks, 
                               max_workers=50, images_per_task_workers=10, image_urls_key=image_urls_key)


def load_jsonl_tasks(*jsonl_paths: str) -> List[Dict]:
    """Load tasks from one or more JSONL files in the Kera production annotation format.

    Each line is expected to have the structure:
        {"image_id": "...", "image_path": "gs://...", "fields": {...}, ...}

    The ``image_path`` GCS URL is wrapped into a list under the ``image_urls`` key
    so the existing download machinery can consume each record unchanged.

    Args:
        *jsonl_paths: Paths to one or more ``.jsonl`` files. Paths that do not
            exist are skipped with a warning.

    Returns:
        Flat list of task dicts with an ``image_urls`` key added.
    """
    output: List[Dict] = []
    for path in jsonl_paths:
        if not path:
            continue
        if not os.path.exists(path):
            print(f"Warning: JSONL file not found: {path}, skipping...")
            continue
        count = 0
        with open(path, encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as exc:
                    print(f"Warning: JSON parse error in {path} line {line_no}: {exc}, skipping...")
                    continue
                gcs_url: str = record.get("image_path", "")
                record["image_urls"] = [gcs_url] if gcs_url else []
                output.append(record)
                count += 1
        print(f"Loaded {count} tasks from {path}")
    return output


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Download images from GCS based on annotated JSONL files.")
    parser.add_argument("--train_jsonl", type=str, required=True,
                        help="Path to the annotated training JSONL file (image_path is a GCS URL).")
    parser.add_argument("--test_jsonl", type=str, default=None,
                        help="Path to the annotated test JSONL file (optional).")
    parser.add_argument("--secrets_path", type=str, required=True,
                        help="Path to Google Cloud service account JSON file.")
    parser.add_argument("--images_output_dir", type=str, required=True,
                        help="Directory to save downloaded images.")
    return parser.parse_args()


def main():
    args = get_args()

    with open(args.secrets_path) as f:
        secrets = json.load(f)
    client = storage.Client.from_service_account_info(secrets)

    tasks = load_jsonl_tasks(args.train_jsonl, args.test_jsonl or "")

    images_bucket_name = "kera-production.appspot.com"
    download_tasks_images_fast(client, images_bucket_name, args.images_output_dir, tasks,
                               image_urls_key="image_urls")

if __name__ == "__main__":
    main()