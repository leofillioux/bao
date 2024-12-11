def dice_loss(y_hat, y):
    y_hat = y_hat.flatten()
    y = y.flatten()
    intersection = (y_hat * y).sum()
    return 1 - ((2. * intersection) / (y_hat.sum() + y.sum()))

def dice_metric(y_hat, y):
    y_hat = y_hat.flatten() > 0.5
    y = y.flatten()
    intersection = (y_hat * y).sum()
    return (2. * intersection) / (y_hat.sum() + y.sum())