import os
import json
from PIL import Image
from google.cloud import storage

from google.api_core.exceptions import NotFound
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import List, Dict, Tuple
import structlog

logger = structlog.get_logger(__name__)

def is_valid_image_file(file_path: str) -> bool:
    
    try:
        image = Image.open(file_path).convert("RGB")
        image.verify()  # Verify that it is, in fact an image
        return True
    except (IOError, SyntaxError) as e:
        logger.error("Invalid image file", file_path=file_path, error=str(e))
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
            logger.warning("Removing corrupted file", file_path=output_path)
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
    logger.info("Starting image download for tasks", total_tasks=len(tasks), max_workers=max_workers, images_per_task_workers=images_per_task_workers)
    
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
                    logger.error("Error processing task", task_id=task_id, error=str(e))
                    with progress_lock:
                        total_failed += len(task.get(image_urls_key, []))
                
                pbar.update(1)
    
    logger.info(f"\n{'='*80}")
    logger.info("DOWNLOAD SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total images: {total_images}")
    logger.info(f"Successfully downloaded: {total_successful}")
    logger.info(f"Failed downloads: {total_failed}")
    logger.info(f"Success rate: {100 * total_successful / max(1, total_images):.1f}%")
    
    if all_failed_urls:
        logger.warning("Failed URLs (showing first 10)", failed_urls=all_failed_urls[:10])
        for url in all_failed_urls[:10]:
            logger.warning("  - {url}", url=url)
        if len(all_failed_urls) > 10:
            logger.warning("  ... and {remaining} more", remaining=len(all_failed_urls) - 10)
    
    logger.info(f"{'='*80}\n")
    
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
    """Load tasks from one or more local JSONL files in the Kera production annotation format.

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
            logger.warning("JSONL file not found, skipping...", path=path)
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
                    logger.warning("JSON parse error, skipping...", path=path, line=line_no, error=str(exc))
                    continue
                gcs_url: str = record.get("image_path", "")
                record["image_urls"] = [gcs_url] if gcs_url else []
                output.append(record)
                count += 1
        logger.info("Loaded tasks from local JSONL", count=count, path=path)
    return output


DEFAULT_ANNOTATIONS_GCS_PREFIX = "annotated_image_data/prescriptions/v_20260402_013946/"


def fetch_annotation_files_from_gcs(
    client: storage.Client,
    bucket_name: str,
    gcs_prefix: str,
    local_dir: str,
) -> List[str]:
    """Download all JSONL annotation files from a GCS prefix to a local directory.

    Lists every blob under ``gcs_prefix`` whose name ends in ``.jsonl``, downloads
    each one to ``local_dir`` (preserving only the filename, not any sub-path), and
    returns the list of local paths that were written.

    Args:
        client:      Authenticated GCS client.
        bucket_name: Name of the GCS bucket.
        gcs_prefix:  Object prefix under which annotation JSONL files live,
                     e.g. ``annotated_image_data/prescriptions/v_20260402_013946/``.
        local_dir:   Local directory where files will be saved.

    Returns:
        List of absolute local file paths for every downloaded JSONL file.
    """
    os.makedirs(local_dir, exist_ok=True)
    bucket = client.bucket(bucket_name)
    blobs = list(client.list_blobs(bucket_name, prefix=gcs_prefix))
    jsonl_blobs = [b for b in blobs if b.name.endswith(".jsonl")]

    if not jsonl_blobs:
        logger.warning(
            "No JSONL files found at GCS prefix",
            bucket=bucket_name,
            prefix=gcs_prefix,
        )
        return []

    logger.info(
        "Found annotation JSONL files",
        count=len(jsonl_blobs),
        bucket=bucket_name,
        prefix=gcs_prefix,
    )

    local_paths: List[str] = []
    for blob in jsonl_blobs:
        filename = os.path.basename(blob.name)
        local_path = os.path.join(local_dir, filename)
        if os.path.exists(local_path):
            logger.info("Annotation file already exists, skipping download", path=local_path)
        else:
            logger.info("Downloading annotation file", blob=blob.name, dest=local_path)
            blob.download_to_filename(local_path)
        local_paths.append(local_path)

    return local_paths


def get_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download prescription images and annotation JSONL files from GCS."
    )
    parser.add_argument(
        "--annotations_gcs_prefix",
        type=str,
        default=DEFAULT_ANNOTATIONS_GCS_PREFIX,
        help=(
            "GCS object prefix under which annotation JSONL files live "
            f"(default: {DEFAULT_ANNOTATIONS_GCS_PREFIX})"
        ),
    )
    parser.add_argument(
        "--annotations_local_dir",
        type=str,
        required=True,
        help="Local directory where annotation JSONL files will be saved after download.",
    )
    parser.add_argument(
        "--secrets_path",
        type=str,
        required=True,
        help="Path to Google Cloud service account JSON file.",
    )
    parser.add_argument(
        "--images_output_dir",
        type=str,
        required=True,
        help="Local directory to save downloaded images.",
    )
    return parser.parse_args()


def main():
    args = get_args()

    with open(args.secrets_path) as f:
        secrets = json.load(f)
    client = storage.Client.from_service_account_info(secrets)

    images_bucket_name = "kera-production.appspot.com"

    # Step 1: download annotation JSONL files from GCS
    local_jsonl_paths = fetch_annotation_files_from_gcs(
        client,
        images_bucket_name,
        args.annotations_gcs_prefix,
        args.annotations_local_dir,
    )

    if not local_jsonl_paths:
        logger.error(
            "No annotation files found; cannot download images.",
            prefix=args.annotations_gcs_prefix,
        )
        raise SystemExit(1)

    # Step 2: load tasks and download images
    tasks = load_jsonl_tasks(*local_jsonl_paths)
    download_tasks_images_fast(
        client,
        images_bucket_name,
        args.images_output_dir,
        tasks,
        image_urls_key="image_urls",
    )

if __name__ == "__main__":
    main()