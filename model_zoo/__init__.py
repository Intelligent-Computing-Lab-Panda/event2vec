from model_zoo import custom
def create_classifier(backbone:str, n_classes):
    if backbone.startswith('DVSGestureNet-'):
        # manual_mlp-{P,H,W}-{n_layers,d_model,d_feedforward,nheads,dropout,activation}
        backbone = backbone.split('-')

        temp = backbone[1].split(',')
        P = int(temp[0])
        H = int(temp[1])
        W = int(temp[2])


        temp = backbone[2].split(',')
        d_model = int(temp[0])
        d_feedforward = int(temp[1])
        nheads = int(temp[2])
        activation = temp[3].lower()

        return custom.DVSGestureNet(P=P, H=H, W=W, d_model=d_model, d_feedforward=d_feedforward, nheads=nheads, n_classes=n_classes, activation=activation)

    elif backbone.startswith('ASLDVSNet-'):
        # manual_mlp-{P,H,W}-{n_layers,d_model,d_feedforward,nheads,dropout,activation}
        backbone = backbone.split('-')

        temp = backbone[1].split(',')
        P = int(temp[0])
        H = int(temp[1])
        W = int(temp[2])


        temp = backbone[2].split(',')
        d_model = int(temp[0])
        d_feedforward = int(temp[1])
        nheads = int(temp[2])
        activation = temp[3].lower()

        return custom.ASLDVSNet(P=P, H=H, W=W, d_model=d_model, d_feedforward=d_feedforward, nheads=nheads, n_classes=n_classes, activation=activation)

    else:
        raise NotImplementedError


if __name__ == '__main__':
    net = create_classifier('ASLDVSNet-2,180,240-64,128,2,relu', n_classes=24)

    n_param = 0
    for p in net.parameters():
        if p.requires_grad:
            n_param += p.numel()

    print('param in MB:', n_param * 4 / 1024 / 1024)