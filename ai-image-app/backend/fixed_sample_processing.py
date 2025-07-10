
def process_sample(sample):
    """Process a LAION-2B sample with proper key handling"""
    try:
        # Normalize keys to lowercase
        sample_lower = {k.lower(): v for k, v in sample.items()}
        
        # Try multiple possible text keys
        text = (sample_lower.get("text") or 
                sample_lower.get("caption") or 
                sample_lower.get("title") or 
                "")
        
        # Try multiple possible URL keys
        url = (sample_lower.get("url") or 
               sample_lower.get("image_url") or 
               sample_lower.get("image") or 
               "")
        
        # Validate
        if not text or not url:
            return None
            
        if len(text) < 5 or len(text) > 200:
            return None
            
        return {"text": text, "url": url}
        
    except Exception as e:
        logger.warning(f"Failed to process sample: {e}")
        return None

# Use this in your training loop:
for sample in dataset:
    processed = process_sample(sample)
    if processed is None:
        continue
    
    # Now you can safely use processed["text"] and processed["url"]
    text = processed["text"]
    url = processed["url"]
