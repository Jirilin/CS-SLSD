#Pseudolabel generation using model's own predictions with confidence thresholding
import torch
import torch.nn.functional as F

class SimplePseudolabelGenerator:
    def __init__(self, model, confidence_threshold=0.9, device='cpu'):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def generate(self, images):
        """
        Generate pseudolabels for a batch of images.
        
        Args:
            images (torch.Tensor): Batch of unlabeled images, shape (B, C, H, W).
        
        Returns:
            confident_images (torch.Tensor): Images that passed the threshold.
            pseudolabels (torch.Tensor): Corresponding pseudolabels.
            mask (torch.BoolTensor): Boolean mask of confident samples.
            probabilities (torch.Tensor): Max probabilities for all images (for analysis).
        """
        with torch.no_grad():
            images = images.to(self.device)
            logits = self.model(images)
            probs = F.softmax(logits, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            mask = max_probs >= self.confidence_threshold
            confident_images = images[mask]
            pseudolabels = preds[mask]
        return confident_images, pseudolabels, mask, max_probs
    
    def update_threshold(self, new_threshold):
        """Allow dynamic threshold adjustment (e.g., curriculum pseudo‑labeling)."""
        self.confidence_threshold = new_threshold
        print(f"Confidence threshold updated to {new_threshold}")
    
    def get_statistics(self, images, true_labels=None):
        """
        Compute statistics: number confident, average confidence, accuracy if true_labels provided.
        """
        _, _, mask, max_probs = self.generate(images)
        num_confident = mask.sum().item()
        avg_conf = max_probs[mask].mean().item() if num_confident > 0 else 0.0
        
        stats = {
            'batch_size': len(images),
            'num_confident': num_confident,
            'ratio_confident': num_confident / len(images),
            'avg_confidence': avg_conf
        }
        if true_labels is not None:
            # For evaluation only – requires true labels
            confident_labels = true_labels[mask]
            _, pseudolabels, _, _ = self.generate(images)
            if num_confident > 0:
                correct = (pseudolabels == confident_labels).sum().item()
                stats['accuracy'] = 100 * correct / num_confident
        return stats

# ------------------- Demonstration -------------------
if __name__ == "__main__":
    from stream_simulator import MNISTStreamSimulator
    from models import SimpleCNN
    
    # Load pre‑trained model
    device = 'cpu'
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load('baseline_offline_model.pth', map_location=device))
    
    # Create generator
    generator = SimplePseudolabelGenerator(model, confidence_threshold=0.9, device=device)
    
    # Simulate stream
    sim = MNISTStreamSimulator(batch_size=64, initial_labeled_size=1000)
    
    # Process 5 batches
    for i in range(5):
        images, _ = sim.next_batch()
        if images is None:
            break
        stats = generator.get_statistics(images)
        print(f"Batch {i}: {stats['num_confident']}/{stats['batch_size']} confident, "
              f"avg confidence = {stats['avg_confidence']:.3f}")