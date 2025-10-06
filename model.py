import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

# =====================================================
# ✅ LSD Model Architecture (original trained structure)
# =====================================================
class LSDProjectionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Hidden projection (LLM hidden states → semantic space)
        self.hidden_proj = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256)
        )

        # Truth projection (sentence transformer → semantic space)
        self.truth_proj = nn.Sequential(
            nn.Linear(384, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256)
        )

    def forward(self, hidden_vec, truth_vec):
        h = self.hidden_proj(hidden_vec)
        t = self.truth_proj(truth_vec)
        h = nn.functional.normalize(h, p=2, dim=-1)
        t = nn.functional.normalize(t, p=2, dim=-1)
        return torch.sum(h * t, dim=-1)  # cosine similarity


# =====================================================
# ✅ Load model + pre-trained projection heads
# =====================================================
def load_trained_model(h_path="h_proj_final.pt", t_path="t_proj_final.pt"):
    model = LSDProjectionModel()
    model.hidden_proj.load_state_dict(torch.load(h_path, map_location="cpu"))
    model.truth_proj.load_state_dict(torch.load(t_path, map_location="cpu"))
    model.eval()
    return model


# =====================================================
# ✅ Inference Example
# =====================================================
if __name__ == "__main__":
    model = load_trained_model()

    # Load sentence transformer (truth encoder)
    truth_encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # Example factual & hallucinated text
    factual = "The Earth revolves around the Sun once every 365 days."
    hallucination = "The Sun revolves around the Earth every 24 hours."

    # Generate embeddings (hidden & truth)
    # For demo, simulate LLM hidden vector as random (replace with your extracted state)
    hidden_factual = torch.randn(1, 768)
    hidden_halluc = torch.randn(1, 768)

    truth_factual = torch.tensor(truth_encoder.encode(factual)).unsqueeze(0)
    truth_halluc = torch.tensor(truth_encoder.encode(hallucination)).unsqueeze(0)

    # Compute semantic alignment
    with torch.no_grad():
        align_fact = model(hidden_factual, truth_factual).item()
        align_hall = model(hidden_halluc, truth_halluc).item()

    print(f"✅ Factual alignment: {align_fact:.3f}")
    print(f"❌ Hallucination alignment: {align_hall:.3f}")
