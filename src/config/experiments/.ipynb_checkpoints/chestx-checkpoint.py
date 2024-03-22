from config.defaults import Experiment, SiMVC, CNN, DDC, Fusion, Loss, Dataset, CoMVC, Optimizer, MLP

chestx = Experiment(
    dataset_config=Dataset(name="chestx"),
    model_config=SiMVC(
        backbone_configs=(
            CNN(input_size=(1, 128, 128)),
            CNN(input_size=(1, 128, 128)),
        ),
        fusion_config=Fusion(method="weighted_mean", n_views=2),
        projector_config=None,
        cm_config=DDC(n_clusters=547),
        loss_config=Loss(
            funcs="ddc_1|ddc_2|ddc_3",
        ),
        optimizer_config=Optimizer(learning_rate=1e-3, clip_norm=10, scheduler_step_size=50),
        use_attention=True
    ),
    n_epochs=100,
    batch_size=8,

)

chestx_contrast = Experiment(
    dataset_config=Dataset(name="chestx"),
    model_config=CoMVC(
        backbone_configs=(
            CNN(input_size=(1, 128, 128)),
            CNN(input_size=(1, 128, 128)),
        ),
        fusion_config=Fusion(method="weighted_mean", n_views=2),
        projector_config=None,
        cm_config=DDC(n_clusters=20),
        loss_config=Loss(
            funcs="ddc_1|ddc_2|ddc_3|contrast",
            delta=20.0
        ),
        optimizer_config=Optimizer()
    ),
    batch_size=4
)
