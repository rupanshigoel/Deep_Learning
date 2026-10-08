"""Print the actual filtered CLEVR-X train dataset size (after recut + factual_explanation filters)."""
from col775_a2.data.clevrx_dataset import ClevrXQADataset
ds = ClevrXQADataset(
    "A2_dataset/Part_B/CLEVR_X/CLEVR_train_explanations_v0.7.10.json",
    "A2_dataset/Part_Aa/Clevr_official/images/train",
    recut_pkl="A2_dataset/Part_B/CLEVR_X/train_images_ids_v0.7.10-recut.pkl",
)
print("FILTERED LEN =", len(ds.entries))
print("700K assumption was off by:", 700000 - len(ds.entries))
print("True full epoch (eff batch 32) =", len(ds.entries) // 32, "optim steps")
