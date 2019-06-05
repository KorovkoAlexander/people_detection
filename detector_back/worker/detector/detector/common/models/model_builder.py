from detector.common.functions.prior_box import PriorBox, forward_features_size
# ssds part
from detector.common.models.ssds import ssd
from detector.common.models.ssds import ssd_lite
from detector.common.models.ssds import rfb
from detector.common.models.ssds import rfb_lite
from detector.common.models.ssds import fssd
from detector.common.models.ssds import fssd_lite
from detector.common.models.ssds import yolo
from detector.common.models.ssds import M2Det

# nets part
from detector.common.models.nets import vgg
from detector.common.models.nets import resnet
from detector.common.models.nets import mobilenet
from detector.common.models.nets import darknet
from detector.common.models.nets import dpn
from detector.common.models.nets import shufflenet
from detector.common.models.nets import pelee

ssds_map = {
                'ssd': ssd.build_ssd,
                'ssd_lite': ssd_lite.build_ssd_lite,
                'rfb': rfb.build_rfb,
                'rfb_lite': rfb_lite.build_rfb_lite,
                'fssd': fssd.build_fssd,
                'fssd_lite': fssd_lite.build_fssd_lite,
                'yolo_v2': yolo.build_yolo_v2,
                'yolo_v3': yolo.build_yolo_v3,
                'm2det': M2Det.build_ssd,
            }

networks_map = {
                    'vgg16': vgg.vgg16,
                    'resnet_18': resnet.resnet_18,
                    'resnet_34': resnet.resnet_34,
                    'resnet_50': resnet.resnet_50,
                    'resnet_101': resnet.resnet_101,
                    'mobilenet_v1': mobilenet.mobilenet_v1,
                    'mobilenet_v1_075': mobilenet.mobilenet_v1_075,
                    'mobilenet_v1_050': mobilenet.mobilenet_v1_050,
                    'mobilenet_v1_025': mobilenet.mobilenet_v1_025,
                    'mobilenet_v2': mobilenet.mobilenet_v2,
                    'mobilenet_v2_075': mobilenet.mobilenet_v2_075,
                    'mobilenet_v2_050': mobilenet.mobilenet_v2_050,
                    'mobilenet_v2_025': mobilenet.mobilenet_v2_025,
                    'darknet_19': darknet.darknet_19,
                    'darknet_53': darknet.darknet_53,
                    'dpn_68': dpn.dpn68,
                    'dpn_68_b': dpn.dpn68b,
                    'dpn_92': dpn.dpn92,
                    'dpn_98': dpn.dpn98,
                    'dpn_107': dpn.dpn107,
                    'dpn_131': dpn.dpn131,
                    'shufflenet': shufflenet.shufflenet,
                    'pelee': pelee.pelee
               }


def create_model(cfg):
    base = networks_map[cfg.NETS]
    number_box = [
        2*len(aspect_ratios)
        if isinstance(aspect_ratios[0], int)
        else len(aspect_ratios)
        for aspect_ratios in cfg.ASPECT_RATIOS
    ]
        
    model = ssds_map[cfg.SSDS](
        base=base,
        feature_layer=cfg.FEATURE_LAYER,
        mbox=number_box,
        num_classes=cfg.NUM_CLASSES
    )
    #
    feature_maps = forward_features_size(model, cfg.IMAGE_SIZE)
    print('==>Feature map size:')
    print(feature_maps)
    #input()
    # 
    priorbox = PriorBox(image_size=cfg.IMAGE_SIZE, feature_maps=feature_maps, aspect_ratios=cfg.ASPECT_RATIOS, 
                    scale=cfg.SIZES, archor_stride=cfg.STEPS, clip=cfg.CLIP)
    # priors = Variable(priorbox.forward(), volatile=True)

    return model, priorbox