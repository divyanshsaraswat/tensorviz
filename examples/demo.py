import torch
import torch.nn as nn
from tensorviz import register_hooks, get_logs, render, print_logs
from tensorviz.reporter import export_json,render_html

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 30 * 30, 10)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleNet()
register_hooks(model)
x = torch.randn(1, 3, 32, 32)
model(x)

logs = get_logs()
print_logs()
render(logs, "simple_net_graph")
export_json(logs)
render_html(logs)
