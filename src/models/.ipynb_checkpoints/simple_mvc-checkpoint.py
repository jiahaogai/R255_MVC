import helpers
from lib.loss import Loss
from lib.fusion import get_fusion_module
from lib.optimizer import Optimizer
from lib.backbones import Backbones
from models.model_base import ModelBase
from models.clustering_module import DDC
from models.attention_module import AttentionModule

class SiMVC(ModelBase):
    def __init__(self, cfg):
        """
        Implementation of the SiMVC model.

        :param cfg: Model config. See `config.defaults.SiMVC` for documentation on the config object.
        """
        super().__init__()

        self.cfg = cfg
        self.output = self.hidden = self.fused = self.backbone_outputs = None

        # Define Backbones and Fusion modules
        self.backbones = Backbones(cfg.backbone_configs)
        self.fusion = get_fusion_module(cfg.fusion_config, self.backbones.output_sizes)
        # Define clustering module
        self.ddc = DDC(input_dim=self.fusion.output_size, cfg=cfg.cm_config)
        # Define loss-module
        self.loss = Loss(cfg=cfg.loss_config)
        # Initialize weights.
        self.apply(helpers.he_init_weights)

        # Instantiate optimizer
        self.optimizer = Optimizer(cfg.optimizer_config, self.parameters())
        # Define attention module
        self.attention = AttentionModule(self.backbones.output_sizes)
        print("init with SiMVC")

    def forward(self, views):
        self.backbone_outputs = self.backbones(views)
        attention_weights = self.attention(self.backbone_outputs)

        # Compute weighted outputs
        weighted_outputs = []
        for output, weight in zip(self.backbone_outputs, attention_weights.split(1, dim=1)):
            weighted_output = output * weight.expand_as(output)
            weighted_outputs.append(weighted_output)
        self.fused = self.fusion(weighted_outputs)
        self.output, self.hidden = self.ddc(self.fused)
        return self.output
