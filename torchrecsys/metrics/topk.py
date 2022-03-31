import torch


class ndcg(torch.nn.Module):
    def __init__(
          self,
          candidates,
          candidate_model=None,
          query_model=None,
          k=50,
      ):
        super().__init__()
        self._k = k
        self._candidates = candidates
        
        
    def _compute_score(self, queries, candidates):
        scores = torch.matmul(
            queries, torch.transpose(candidates,0,1)
        )
        return scores
    
    def forward(self, queries):
        
        if self.query_model is not None:
            queries = self.query_model(queries)
        
        if self.candidate_model is not None:
            self._candidates = self.candidate_model(self._candidates)
        
        scores = self._compute_score(queries, self._candidates)
        
        values, indices = torch.topk(scores, k=self.k)

        return values, indices
        