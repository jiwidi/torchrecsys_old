import torch

class Ranking():
    def __init__(self):
        self.loss = torch.nn.CrossEntropyLoss(reduction="sum")
        pass
    
    def __call__(self, query_embeddings, candidate_embeddings):
        scores = torch.matmul(
            query_embeddings, torch.transpose(candidate_embeddings,0,1)
        )
        
        num_queries, num_candidates = scores.shape

        labels = torch.range(0, num_queries-1, dtype=int)
    
    
        loss = self.loss(input=scores, target=labels)
        
        return loss
