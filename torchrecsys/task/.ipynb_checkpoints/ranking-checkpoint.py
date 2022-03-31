import torch
from torchmetrics import RetrievalNormalizedDCG


class Ranking():
    def __init__(self, metric=None):
        self.loss = torch.nn.CrossEntropyLoss(reduction="sum")
        self.metric = metric
        pass
    
    def __call__(self, query_embeddings, candidate_embeddings):
        scores = torch.matmul(
            query_embeddings, candidate_embeddings.t()
        )
        
        num_queries, num_candidates = scores.shape

        labels = torch.arange(0, num_queries, dtype=int, device=scores.device)
    
        
        loss = self.loss(input=scores, target=labels)
        if self.metric is not None:
            r = self.metric(query_embeddings, candidate_embeddings)
            
            
        return loss
