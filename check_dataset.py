import event_dataset
from event_dataset.dvs_lip import cluster_dvs_lip

def check_dvs_gesture_dataset(root = '/dev/shm/DVS128Gesture/events_np'):
    dts = event_dataset.EventDataModule(name='dvs_gesture', root=root, sample_number=1024, sampler='random_sample', batch_size=64, num_workers=8)

    dts.setup('')
    train_loader = dts.train_dataloader()
    val_loader = dts.val_dataloader()
    for (x, y, t, p), intensity, valid_mask, label in train_loader:
        print(x[0], y[0], t[0], p[0])
        print(intensity, valid_mask[0], label[0])
        break
    for (x, y, t, p), intensity, valid_mask, label, indices in val_loader:
        print(x[0], y[0], t[0], p[0])
        print(intensity, valid_mask[0], label[0], indices[0])
        break

def check_asl_dvs_dataset(root = '/dev/shm/ASLDVS/events_np'):
    dts = event_dataset.EventDataModule(name='asl_dvs', root=root, sample_number=1024, sampler='random_sample', batch_size=64, num_workers=8)

    dts.setup('')
    train_loader = dts.train_dataloader()
    val_loader = dts.val_dataloader()
    for (x, y, t, p), intensity, valid_mask, label in train_loader:
        print(x[0], y[0], t[0], p[0])
        print(intensity, valid_mask[0], label[0])
        break
    for (x, y, t, p), intensity, valid_mask, label, indices in val_loader:
        print(x[0], y[0], t[0], p[0])
        print(intensity, valid_mask[0], label[0], indices[0])
        break

def check_dvs_lip_dataset(root = '/dev/shm/dvs_lip'):
    dts = event_dataset.EventDataModule(name='dvs_lip', root=root, sample_number=1024, sampler='random_sample', batch_size=64, num_workers=8)

    dts.setup('')
    train_loader = dts.train_dataloader()
    val_loader = dts.val_dataloader()
    for (x, y, t, p), intensity, valid_mask, label in train_loader:
        print(x[0], y[0], t[0], p[0])
        print(intensity, valid_mask[0], label[0])
        break
    for (x, y, t, p), intensity, valid_mask, label, indices in val_loader:
        print(x[0], y[0], t[0], p[0])
        print(intensity, valid_mask[0], label[0], indices[0])
        break
if __name__ == '__main__':
    
    # check_dvs_gesture_dataset()
    # check_asl_dvs_dataset()
    # check_dvs_lip_dataset()
    cluster_dvs_lip(out_dir='/dev/shm/dvs_lip/kmeans_1024', save=True)

    