import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import random
import numpy as np
from src.models.evflownet import EVFlowNet
from src.datasets import DatasetProvider
from enum import Enum, auto
from src.datasets import train_collate
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any
import os
import time

class RepresentationType(Enum):
    VOXEL = auto()
    STEPAN = auto()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def compute_multiscale_epe_error(pred_flows: Dict[str, torch.Tensor], gt_flow: torch.Tensor):
    total_loss = 0.0
    for scale, pred_flow in pred_flows.items():
        scale_factor = pred_flow.shape[2] / gt_flow.shape[2]
        scaled_gt_flow = torch.nn.functional.interpolate(gt_flow, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        epe = torch.mean(torch.norm(pred_flow - scaled_gt_flow, p=2, dim=1))
        total_loss += epe
    return total_loss

def save_optical_flow_to_npy(flow: torch.Tensor, file_name: str):
    np.save(f"{file_name}.npy", flow.cpu().numpy())

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(args: DictConfig):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    loader = DatasetProvider(
        dataset_path=Path(args.dataset_path),
        representation_type=RepresentationType.VOXEL,
        delta_t_ms=100,
        num_bins=4
    )
    train_set = loader.get_train_dataset()
    test_set = loader.get_test_dataset()
    collate_fn = train_collate
    train_data = DataLoader(train_set,
                                 batch_size=args.data_loader.train.batch_size,
                                 shuffle=args.data_loader.train.shuffle,
                                 collate_fn=collate_fn,
                                 drop_last=False)
    test_data = DataLoader(test_set,
                                 batch_size=args.data_loader.test.batch_size,
                                 shuffle=args.data_loader.test.shuffle,
                                 collate_fn=collate_fn,
                                 drop_last=False)

    model = EVFlowNet(args.train).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.train.initial_learning_rate, weight_decay=args.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model.train()
    for epoch in range(args.train.epochs):
        total_loss = 0
        print("on epoch: {}".format(epoch+1))
        for i, batch in enumerate(tqdm(train_data)):
            batch: Dict[str, Any]
            event_image = batch["event_volume"].to(device)  # [B, 8, 480, 640]  変更点: 2フレーム分のチャネル
            ground_truth_flow = batch["flow_gt"].to(device)  # [B, 2, 480, 640]
            flow_dict = model(event_image)  # モデルの出力を辞書として取得
            loss: torch.Tensor = compute_multiscale_epe_error(flow_dict, ground_truth_flow)  # 新しい損失関数を使用
            print(f"batch {i} loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_data)}')

        scheduler.step()

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    current_time = time.strftime("%Y%m%d%H%M%S")
    model_path = f"checkpoints/model_{current_time}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    flow: torch.Tensor = torch.tensor([]).to(device)
    with torch.no_grad():
        print("start test")
        for batch in tqdm(test_data):
            batch: Dict[str, Any]
            event_image = batch["event_volume"].to(device)
            batch_flow = model(event_image)["flow3"]  # 最後のスケールの出力を使用
            flow = torch.cat((flow, batch_flow), dim=0)  # [N, 2, 480, 640]
        print("test done")
    
    file_name = "submission"
    save_optical_flow_to_npy(flow, file_name)

if __name__ == "__main__":
    main()
