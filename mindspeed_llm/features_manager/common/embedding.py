from argparse import ArgumentParser
from mindspeed.features_manager.feature import MindSpeedFeature


class LanguageModelEmbeddingFeature(MindSpeedFeature):
    def __init__(self):
        super(LanguageModelEmbeddingFeature, self).__init__(feature_name="language-model-embedding", optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument(
            "--hidden-size-per-layer-input",
            type=int,
            default=0,
            help="Per-layer embedding hidden size (PLE). 0 disables PLE.",
        )
        group.add_argument(
            "--vocab-size-per-layer-input",
            type=int,
            default=None,
            help="Vocabulary size for per-layer embeddings (PLE). Defaults to vocab-size when unset.",
        )
        group.add_argument(
            "--ple-alpha",
            type=float,
            default=0.1,
            help="Residual scaling factor for PLE injection. Smaller values make PLE start gentler.",
        )
        group.add_argument(
            "--meki-dim",
            type=int,
            default=0,
            help="MeKi memory dimension. 0 disables MeKi.",
        )
        group.add_argument(
            "--meki-alpha",
            type=float,
            default=1.0,
            help="Residual scaling factor for MeKi branch output.",
        )
        group.add_argument(
            "--meki-beta",
            type=float,
            default=1.0,
            help="Scaling factor for MeKi context projection before fusion with memory lookup.",
        )

    def register_patches(self, patch_manager, args):
        from mindspeed.core.models.common.embeddings.language_model_embedding import language_model_embedding_forward_wrapper
        from mindspeed_llm.core.models.common.embeddings.language_model_embedding import language_model_embedding_init_func

        patch_manager.register_patch('megatron.core.models.common.embeddings.language_model_embedding.LanguageModelEmbedding.__init__',
                                      language_model_embedding_init_func)
