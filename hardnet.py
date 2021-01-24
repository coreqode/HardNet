import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    GlobalAveragePooling2D,
    Dropout,
    Layer,
    UpSampling2D,
    MaxPool2D,
)
from tensorflow.keras.layers import (
    ReLU,
    BatchNormalization,
    Add,
    Reshape,
    DepthwiseConv2D,
    Lambda,
    Dense,
    Concatenate,
)


class ConvBlock(Layer):
    def __init__(self, out_channels, kernel_size=3, stride=1, dropout=0.1, bias=False):
        super().__init__()
        self.conv_block = Sequential(
            [
                Conv2D(
                    out_channels,
                    kernel_size,
                    strides=stride,
                    padding="same",
                    use_bias=bias,
                ),
                BatchNormalization(),
                ReLU(),
            ]
        )

    def call(self, x):
        return self.conv_block(x)


def hardnet68(x, depth_wise=False, arch=68, pretrained=False, weight_path=""):
    first_channel = [32, 64]
    max_pool = True
    m = 1.7
    drop_rate = 0.1

    # HarDNet68
    channel_list = [128, 256, 320, 640, 1024]
    k = [14, 16, 20, 40, 160]
    n_layers = [8, 16, 16, 16, 4]
    down_sample = [1, 0, 1, 1, 0]

    if arch==85:
          #HarDNet85
          first_ch  = [48, 96]
          ch_list = [  192, 256, 320, 480, 720, 1280]
          gr       = [  24,  24,  28,  36,  48, 256]
          n_layers = [   8,  16,  16,  16,  16,   4]
          downSamp = [   1,   0,   1,   0,   1,   0]
          drop_rate = 0.2

    n_blocks = len(n_layers)

    x = ConvBlock(first_channel[0], stride=2)(x)
    x = ConvBlock(first_channel[1], stride=1)(x)
    if max_pool:
        x = MaxPool2D()(x)
    else:
        pass  # depthwise addision

    ## HardBlocks
    ch = first_channel[1]
    for i in range(n_blocks):
        x = HardBlock(ch, k[i], m, n_layers[i], i)(x)
        x = ConvBlock(channel_list[i], kernel_size = 1 )(x)
        ch = channel_list[i]

        if down_sample[i] == 1:
            if max_pool:
                x = MaxPool2D()(x)
            else:
                pass

    x = GlobalAveragePooling2D()(x)
    x = Dropout(drop_rate)(x)
    x = Dense(1000)(x)
    return x


class HardBlock(Layer):
    def __init__(self, in_channels, k, m, n_layers, block_id, keep_base=False):
        super().__init__()
        self.links = []
        self.keep_base = keep_base
        layers_ = []
        for i in range(n_layers):
            outch, inch, link = self.get_link(i + 1, in_channels, k, m)
            self.links.append(link)
            layers_.append(ConvBlock(outch))

        self.layers = layers_

    def call(self, x):
        layers_ = [x]
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = Concatenate(axis = -1)(tin)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)

        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keep_base) or (i == t - 1) or (i % 2 == 1):
                out_.append(layers_[i])
        out = Concatenate(axis = -1)(out_)
        return out

    def get_link(self, layer, base_ch, k, m):
        if layer == 0:
            return base_ch, 0, []
        out_channels = k
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= m
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, k, m)
            in_channels += ch
        return out_channels, in_channels, link


if __name__ == "__main__":
    inp = Input([224, 224, 3])
    out = hardnet68(inp)
    model = Model(inp, out)
    print(model.summary())
    # print(links)
