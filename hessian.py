import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.autograd.functional as AF

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # input_dim = 1, output_dim = 1

    def forward(self, x):
        return self.linear(x)

model = LinearModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

x_train = torch.randn(100, 1)  # 100개의 랜덤 샘플
y_train = 2 * x_train + 3 + 0.1 * torch.randn(100, 1)  # y = 2x + 3에 노이즈 추가

def compute_hessian_eigenvalues(model, loss):
    # model.parameters()를 튜플로 변환
    params = tuple(model.parameters())
    
    # Hessian 계산
    hessian = AF.hessian(lambda *params: loss, params)
    
    # Hessian을 square matrix로 변환
    hessian_matrix = torch.zeros(sum(p.numel() for p in params), sum(p.numel() for p in params))
    offset = 0
    for i, p_i in enumerate(params):
        for j, p_j in enumerate(params):
      
            hessian_block = hessian[i][j].reshape(p_i.numel(), p_j.numel())
            hessian_matrix[offset:offset + p_i.numel(), offset:offset + p_j.numel()] = hessian_block
        offset += p_i.numel()
    
    # eigenvalue 계산 (실수 부분만 사용)
    eigenvalues = torch.linalg.eigvals(hessian_matrix)
    real_eigenvalues = eigenvalues.real  # 실수 부분만 추출
    
    return real_eigenvalues

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    predictions = model(x_train)
    loss = loss_fn(predictions, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

final_predictions = model(x_train)
final_loss = loss_fn(final_predictions, y_train)
eigenvalues = compute_hessian_eigenvalues(model, final_loss)

eigenvalues_np = eigenvalues.detach().numpy()  # Tensor를 NumPy로 변환
plt.hist(eigenvalues_np, bins=20, alpha=0.7, color='blue')
plt.title("Hessian Eigenvalue Distribution")
plt.xlabel("Eigenvalue")
plt.ylabel("Frequency")
plt.show()