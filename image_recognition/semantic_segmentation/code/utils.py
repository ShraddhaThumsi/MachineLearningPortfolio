import task

smooth = 1e-5
def iou(y_true, y_pred):

    y_pred_ = y_pred > 0.5
    y_true_ = y_true > 0.5
    intersection = (y_pred_ & y_true_).sum()
    union = (y_pred_ | y_true_).sum()
    return (intersection + smooth) / (union + smooth)

def dice_coef(y_true, y_pred):
    y_true_= task.K.flatten(y_true)
    y_pred_ = task.K.flatten(y_pred)
    intersection = task.K.sum(y_pred_*y_true_)
    return (2. * intersection + smooth) / (task.K.sum(y_pred_)+task.K.sum(y_true_)+smooth)


def dice_coef_loss(y_true, y_pred):
    return 0.5 * (task.binary_cross_entropy(y_true, y_pred)) - dice_coef(y_true, y_pred)
