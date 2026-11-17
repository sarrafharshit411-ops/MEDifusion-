"""
MediFusion AI - Automated Remaining GPU Training Queue
Trains blood_cell, skin_cancer, and retinopathy models sequentially on GPU with AMP,
then runs full evaluation across all models.
"""

import os
import sys
import time
import logging

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.utils.helpers import load_config, setup_logging
from src.training.train_multi_image import train_single_image_model
from src.training.evaluate import run_evaluation

logger = logging.getLogger(__name__)


def run_queue(datasets=None):
    setup_logging()
    if datasets is None:
        datasets = ["blood_cell", "skin_cancer", "retinopathy"]

    config = load_config(os.path.join(project_root, "configs", "config.yaml"))
    
    logger.info("=" * 70)
    logger.info("STARTING GPU TRAINING QUEUE FOR REMAINING MODELS")
    logger.info(f"Target datasets: {datasets}")
    logger.info("=" * 70)

    results = {}
    for idx, key in enumerate(datasets, 1):
        logger.info(f"\n{'#' * 70}")
        logger.info(f"[{idx}/{len(datasets)}] TRAINING {key.upper()} ON GPU")
        logger.info(f"{'#' * 70}")
        t0 = time.time()
        try:
            res = train_single_image_model(key, config)
            results[key] = res
            logger.info(f"✓ {key} model complete in {time.time() - t0:.1f}s | Test Acc: {res.get('test_accuracy', 0):.4f}")
        except Exception as e:
            logger.error(f"✗ Failed to train {key}: {e}", exc_info=True)
            results[key] = {"error": str(e)}

    logger.info("\n" + "=" * 70)
    logger.info("ALL TRAINING COMPLETE — RUNNING FULL EVALUATION SUITE")
    logger.info("=" * 70)
    try:
        run_evaluation()
        logger.info("✓ Evaluation suite completed successfully!")
    except Exception as e:
        logger.error(f"Evaluation error: {e}", exc_info=True)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None)
    args = parser.parse_args()
    run_queue(args.datasets)
