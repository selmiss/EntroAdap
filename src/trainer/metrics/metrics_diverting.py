"""Metrics dispatcher for routing to different metric computation functions."""


def compute_metrics(eval_metrics, predictions, labels, tokenizer, metric_key_prefix="eval", categories=None, prompts=None):
    """
    Dispatch to appropriate metrics computation based on eval_metrics setting.
    
    Args:
        eval_metrics: Type of metrics to compute ('text', 'qa', or 'none')
        predictions: List of predicted token ID arrays
        labels: List of ground truth token ID arrays
        tokenizer: Tokenizer for decoding
        metric_key_prefix: Prefix for metric keys in output
        categories: Optional list of category/task names for each sample
        prompts: Optional list of prompt token ID arrays
    
    Returns:
        Tuple of (metrics_dict, detailed_results) where detailed_results can be None
    """
    if eval_metrics == "none" or not predictions:
        return {}, None
    
    try:
        detailed_results = None
        
        if eval_metrics == "text":
            from src.trainer.metrics.text_overlap import compute_metrics_text_overlap
            metrics = compute_metrics_text_overlap(predictions, labels, tokenizer)
            
            print(f"\n{'='*60}")
            print(f"Text Generation Metrics ({metric_key_prefix}):")
            print(f"{'='*60}")
            for key, value in metrics.items():
                if key != 'loss':
                    print(f"{key:15s}: {value:.4f}")
            print(f"{'='*60}\n")
            
        elif eval_metrics == "qa":
            from src.trainer.metrics.multiple_choices import compute_metrics_multiple_choice_detailed
            metrics, detailed_results = compute_metrics_multiple_choice_detailed(predictions, labels, tokenizer, categories, prompts)
            
            print(f"\n{'='*70}")
            print(f"Multiple Choice QA Metrics ({metric_key_prefix}):")
            print(f"{'='*70}")
            
            # Display per-category results if available
            category_metrics = {k: v for k, v in metrics.items() if k.startswith('accuracy_')}
            if category_metrics:
                print("\nResults by Category:")
                print(f"{'-'*70}")
                print(f"{'Category':<25} {'Correct':<10} {'Total':<10} {'Accuracy':<15}")
                print(f"{'-'*70}")
                for key in sorted(category_metrics.keys()):
                    category = key.replace('accuracy_', '')
                    accuracy = metrics[f'accuracy_{category}']
                    correct = metrics[f'correct_{category}']
                    total = metrics[f'total_{category}']
                    accuracy_pct = f"{accuracy*100:.2f}%"
                    print(f"{category:<25} {correct:<10} {total:<10} {accuracy_pct:<15}")
                print(f"{'-'*70}")
            
            # Display overall results
            overall_accuracy_pct = f"{metrics['accuracy']*100:.2f}%"
            print(f"\n{'Overall':<25} {metrics['correct']:<10} {metrics['total']:<10} {overall_accuracy_pct:<15}")
            print(f"{'='*70}\n")
        else:
            metrics = {}
        
        return metrics, detailed_results
        
    except Exception as e:
        import traceback
        print(f"Warning: Failed to compute {eval_metrics} metrics: {e}")
        traceback.print_exc()
        return {}, None

