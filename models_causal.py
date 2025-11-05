import torch.nn as nn
import torch


class AdultCensusModel(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.layer1 = nn.Linear(in_features=34, out_features=64)
        self.layer2 = nn.Linear(in_features=64, out_features=128)
        self.layer3 = nn.Linear(in_features=128, out_features=256)
        self.layer4 = nn.Linear(in_features=256, out_features=256)
        self.layer5 = nn.Linear(in_features=256, out_features=256)
        self.layer6 = nn.Linear(in_features=256, out_features=256)
        self.layer7 = nn.Linear(in_features=256, out_features=128)
        self.layer8 = nn.Linear(in_features=128, out_features=64)
        self.layer9 = nn.Linear(in_features=64, out_features=1)
        self.dropout = nn.Dropout(p=p)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu6 = nn.ReLU()
        self.relu7 = nn.ReLU()
        self.relu8 = nn.ReLU()

        self.hooks = []
        self.hidden_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.hidden_outputs = []

        x = self.layer1(x)
        self.hidden_outputs.append(x)
        x = self.dropout(self.relu1(x))
        x = self.layer2(x)
        self.hidden_outputs.append(x)
        x = self.dropout(self.relu2(x))
        x = self.layer3(x)
        self.hidden_outputs.append(x)
        x = self.dropout(self.relu3(x))
        x = self.layer4(x)
        self.hidden_outputs.append(x)
        x = self.dropout(self.relu4(x))
        x = self.layer5(x)
        self.hidden_outputs.append(x)
        x = self.dropout(self.relu5(x))
        x = self.layer6(x)
        self.hidden_outputs.append(x)
        x = self.dropout(self.relu6(x))
        x = self.layer7(x)
        self.hidden_outputs.append(x)
        x = self.dropout(self.relu7(x))
        x = self.layer8(x)
        self.hidden_outputs.append(x)
        x = self.dropout(self.relu8(x))
        x = self.layer9(x)
        return x

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     self.hidden_outputs = []
    #     x = self.dropout(self.relu1(self.layer1(x)))
    #     self.hidden_outputs.append(x)
    #     x = self.dropout(self.relu2(self.layer2(x)))
    #     self.hidden_outputs.append(x)
    #     x = self.dropout(self.relu3(self.layer3(x)))
    #     self.hidden_outputs.append(x)
    #     x = self.dropout(self.relu4(self.layer4(x)))
    #     self.hidden_outputs.append(x)
    #     x = self.dropout(self.relu5(self.layer5(x)))
    #     self.hidden_outputs.append(x)
    #     x = self.dropout(self.relu6(self.layer6(x)))
    #     self.hidden_outputs.append(x)
    #     x = self.dropout(self.relu7(self.layer7(x)))
    #     self.hidden_outputs.append(x)
    #     x = self.dropout(self.relu8(self.layer8(x)))
    #     self.hidden_outputs.append(x)
    #     x = self.layer9(x)
    #     return x

    def register_scaling_hooks(self, candidate_neurons, scaling_factors):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

        for (layer_name, idx), scaling_factor in zip(candidate_neurons, scaling_factors):
            # 对激活值进行缩放
            # layer_num = layer_name.lstrip('layer')
            # relu_layer_name = f"relu{layer_num}"
            # relu_layer = getattr(self, relu_layer_name)

            # 对输出值进行缩放
            layer = getattr(self, layer_name)

            def hook_fn(module, input, output, idx=idx, scaling_factor=scaling_factor):
                output[:, idx] *= (1 + scaling_factor)
                return output

            hook = layer.register_forward_hook(hook_fn)
            self.hooks.append(hook)

    def register_do_hooks(self, layer_name, neuron_idx, do_value):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

        #对激活值进行干预
        # layer_num = layer_name.lstrip('layer')
        # relu_layer_name = f"relu{layer_num}"
        # relu_layer = getattr(self, relu_layer_name)

        #对输出值进行干预
        layer = getattr(self, layer_name)

        def hook_fn(module, input, output, neuron_idx=neuron_idx, do_value=do_value):
            output[:, neuron_idx] = do_value
            return output

        # hook = relu_layer.register_forward_hook(hook_fn)
        hook = layer.register_forward_hook(hook_fn)
        self.hooks.append(hook)


class BankModel(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.layer1 = nn.Linear(in_features=32, out_features=128)
        self.layer2 = nn.Linear(in_features=128, out_features=128)
        self.layer3 = nn.Linear(in_features=128, out_features=128)
        self.layer4 = nn.Linear(in_features=128, out_features=128)
        self.layer5 = nn.Linear(in_features=128, out_features=128)
        self.layer6 = nn.Linear(in_features=128, out_features=128)
        self.layer7 = nn.Linear(in_features=128, out_features=128)
        self.layer8 = nn.Linear(in_features=128, out_features=1)
        self.dropout = nn.Dropout(p=p)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu6 = nn.ReLU()
        self.relu7 = nn.ReLU()
        self.relu8 = nn.ReLU()

        self.hidden_outputs = []
        self.hooks = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.hidden_outputs = []

        x = self.layer1(x)
        self.hidden_outputs.append(x)
        x = self.dropout(self.relu1(x))
        x = self.layer2(x)
        self.hidden_outputs.append(x)
        x = self.dropout(self.relu2(x))
        x = self.layer3(x)
        self.hidden_outputs.append(x)
        x = self.dropout(self.relu3(x))
        x = self.layer4(x)
        self.hidden_outputs.append(x)
        x = self.dropout(self.relu4(x))
        x = self.layer5(x)
        self.hidden_outputs.append(x)
        x = self.dropout(self.relu5(x))
        x = self.layer6(x)
        self.hidden_outputs.append(x)
        x = self.dropout(self.relu6(x))
        x = self.layer7(x)
        self.hidden_outputs.append(x)
        x = self.dropout(self.relu7(x))
        x = self.layer8(x)
        return x

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     self.hidden_outputs = []
    #     x = self.dropout(self.relu1(self.layer1(x)))
    #     self.hidden_outputs.append(x)
    #     x = self.dropout(self.relu2(self.layer2(x)))
    #     self.hidden_outputs.append(x)
    #     x = self.dropout(self.relu3(self.layer3(x)))
    #     self.hidden_outputs.append(x)
    #     x = self.dropout(self.relu4(self.layer4(x)))
    #     self.hidden_outputs.append(x)
    #     x = self.dropout(self.relu5(self.layer5(x)))
    #     self.hidden_outputs.append(x)
    #     x = self.dropout(self.relu6(self.layer6(x)))
    #     self.hidden_outputs.append(x)
    #     x = self.dropout(self.relu7(self.layer7(x)))
    #     self.hidden_outputs.append(x)
    #     x = self.layer8(x)
    #     return x

    def register_scaling_hooks(self, candidate_neurons, scaling_factors):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

        for (layer_name, idx), scaling_factor in zip(candidate_neurons, scaling_factors):
            # 对激活值进行缩放
            # layer_num = layer_name.lstrip('layer')
            # relu_layer_name = f"relu{layer_num}"
            # relu_layer = getattr(self, relu_layer_name)

            # 对输出值进行缩放
            layer = getattr(self, layer_name)

            def hook_fn(module, input, output, idx=idx, scaling_factor=scaling_factor):
                output[:, idx] *= (1 + scaling_factor)
                return output

            hook = layer.register_forward_hook(hook_fn)
            self.hooks.append(hook)

    def register_do_hooks(self, layer_name, neuron_idx, do_value):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

        #对激活值进行干预
        # layer_num = layer_name.lstrip('layer')
        # relu_layer_name = f"relu{layer_num}"
        # relu_layer = getattr(self, relu_layer_name)

        #对输出值进行干预
        layer = getattr(self, layer_name)

        def hook_fn(module, input, output, neuron_idx=neuron_idx, do_value=do_value):
            output[:, neuron_idx] = do_value
            return output

        # hook = relu_layer.register_forward_hook(hook_fn)
        hook = layer.register_forward_hook(hook_fn)
        self.hooks.append(hook)


class DefaultModel(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.layer1 = nn.Linear(in_features=30, out_features=64)
        self.layer2 = nn.Linear(in_features=64, out_features=32)
        self.layer3 = nn.Linear(in_features=32, out_features=16)
        self.layer4 = nn.Linear(in_features=16, out_features=8)
        self.layer5 = nn.Linear(in_features=8, out_features=4)
        self.layer6 = nn.Linear(in_features=4, out_features=1)
        self.dropout = nn.Dropout(p=p)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.hooks = []
        self.hidden_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.hidden_outputs = []

        x = self.layer1(x)
        self.hidden_outputs.append(x)
        x = self.dropout(self.relu1(x))
        x = self.layer2(x)
        self.hidden_outputs.append(x)
        x = self.dropout(self.relu2(x))
        x = self.layer3(x)
        self.hidden_outputs.append(x)
        x = self.dropout(self.relu3(x))
        x = self.layer4(x)
        self.hidden_outputs.append(x)
        x = self.dropout(self.relu4(x))
        x = self.layer5(x)
        self.hidden_outputs.append(x)
        x = self.dropout(self.relu5(x))
        x = self.layer6(x)
        return x

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     self.hidden_outputs = []
    #     x = self.dropout(self.relu1(self.layer1(x)))
    #     self.hidden_outputs.append(x)
    #     x = self.dropout(self.relu2(self.layer2(x)))
    #     self.hidden_outputs.append(x)
    #     x = self.dropout(self.relu3(self.layer3(x)))
    #     self.hidden_outputs.append(x)
    #     x = self.dropout(self.relu4(self.layer4(x)))
    #     self.hidden_outputs.append(x)
    #     x = self.dropout(self.relu5(self.layer5(x)))
    #     self.hidden_outputs.append(x)
    #     x = self.layer6(x)
    #     return x

    def register_scaling_hooks(self, candidate_neurons, scaling_factors):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

        for (layer_name, idx), scaling_factor in zip(candidate_neurons, scaling_factors):
            # 对激活值进行缩放
            # layer_num = layer_name.lstrip('layer')
            # relu_layer_name = f"relu{layer_num}"
            # relu_layer = getattr(self, relu_layer_name)

            # 对输出值进行缩放
            layer = getattr(self, layer_name)

            def hook_fn(module, input, output, idx=idx, scaling_factor=scaling_factor):
                output[:, idx] *= (1 + scaling_factor)
                return output

            hook = layer.register_forward_hook(hook_fn)
            self.hooks.append(hook)

    def register_do_hooks(self, layer_name, neuron_idx, do_value):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

        #对激活值进行干预
        # layer_num = layer_name.lstrip('layer')
        # relu_layer_name = f"relu{layer_num}"
        # relu_layer = getattr(self, relu_layer_name)

        #对输出值进行干预
        layer = getattr(self, layer_name)

        def hook_fn(module, input, output, neuron_idx=neuron_idx, do_value=do_value):
            output[:, neuron_idx] = do_value
            return output

        # hook = relu_layer.register_forward_hook(hook_fn)
        hook = layer.register_forward_hook(hook_fn)
        self.hooks.append(hook)


class CompasModel(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.layer1 = nn.Linear(in_features=12, out_features=64)
        self.layer2 = nn.Linear(in_features=64, out_features=32)
        self.layer3 = nn.Linear(in_features=32, out_features=16)
        self.layer4 = nn.Linear(in_features=16, out_features=8)
        self.layer5 = nn.Linear(in_features=8, out_features=4)
        self.layer6 = nn.Linear(in_features=4, out_features=1)
        self.dropout = nn.Dropout(p=p)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.hooks = []
        self.hidden_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.hidden_outputs = []

        x = self.layer1(x)
        self.hidden_outputs.append(x)
        x = self.dropout(self.relu1(x))
        x = self.layer2(x)
        self.hidden_outputs.append(x)
        x = self.dropout(self.relu2(x))
        x = self.layer3(x)
        self.hidden_outputs.append(x)
        x = self.dropout(self.relu3(x))
        x = self.layer4(x)
        self.hidden_outputs.append(x)
        x = self.dropout(self.relu4(x))
        x = self.layer5(x)
        self.hidden_outputs.append(x)
        x = self.dropout(self.relu5(x))
        x = self.layer6(x)
        return x

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     self.hidden_outputs = []
    #     x = self.dropout(self.relu1(self.layer1(x)))
    #     self.hidden_outputs.append(x)
    #     x = self.dropout(self.relu2(self.layer2(x)))
    #     self.hidden_outputs.append(x)
    #     x = self.dropout(self.relu3(self.layer3(x)))
    #     self.hidden_outputs.append(x)
    #     x = self.dropout(self.relu4(self.layer4(x)))
    #     self.hidden_outputs.append(x)
    #     x = self.dropout(self.relu5(self.layer5(x)))
    #     self.hidden_outputs.append(x)
    #     x = self.layer6(x)
    #     return x

    def register_scaling_hooks(self, candidate_neurons, scaling_factors):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

        for (layer_name, idx), scaling_factor in zip(candidate_neurons, scaling_factors):
            # 对激活值进行缩放
            # layer_num = layer_name.lstrip('layer')
            # relu_layer_name = f"relu{layer_num}"
            # relu_layer = getattr(self, relu_layer_name)

            # 对输出值进行缩放
            layer = getattr(self, layer_name)

            def hook_fn(module, input, output, idx=idx, scaling_factor=scaling_factor):
                output[:, idx] *= (1 + scaling_factor)
                return output

            hook = layer.register_forward_hook(hook_fn)
            self.hooks.append(hook)

    def register_do_hooks(self, layer_name, neuron_idx, do_value):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

        #对激活值进行干预
        # layer_num = layer_name.lstrip('layer')
        # relu_layer_name = f"relu{layer_num}"
        # relu_layer = getattr(self, relu_layer_name)

        #对输出值进行干预
        layer = getattr(self, layer_name)

        def hook_fn(module, input, output, neuron_idx=neuron_idx, do_value=do_value):
            output[:, neuron_idx] = do_value
            return output

        # hook = relu_layer.register_forward_hook(hook_fn)
        hook = layer.register_forward_hook(hook_fn)
        self.hooks.append(hook)


class MEPS16Model(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.layer1 = nn.Linear(in_features=138, out_features=128)
        self.layer2 = nn.Linear(in_features=128, out_features=128)
        self.layer3 = nn.Linear(in_features=128, out_features=128)
        self.layer4 = nn.Linear(in_features=128, out_features=1)
        self.dropout = nn.Dropout(p=p)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        self.hooks = []
        self.hidden_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.hidden_outputs = []

        x = self.layer1(x)
        self.hidden_outputs.append(x)
        x = self.dropout(self.relu1(x))
        x = self.layer2(x)
        self.hidden_outputs.append(x)
        x = self.dropout(self.relu2(x))
        x = self.layer3(x)
        self.hidden_outputs.append(x)
        x = self.dropout(self.relu3(x))
        x = self.layer4(x)
        return x

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     self.hidden_outputs = []
    #     x = self.dropout(self.relu1(self.layer1(x)))
    #     self.hidden_outputs.append(x)
    #     x = self.dropout(self.relu2(self.layer2(x)))
    #     self.hidden_outputs.append(x)
    #     x = self.dropout(self.relu3(self.layer3(x)))
    #     self.hidden_outputs.append(x)
    #     x = self.layer4(x)
    #     return x

    def register_scaling_hooks(self, candidate_neurons, scaling_factors):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

        for (layer_name, idx), scaling_factor in zip(candidate_neurons, scaling_factors):
            # 对激活值进行缩放
            # layer_num = layer_name.lstrip('layer')
            # relu_layer_name = f"relu{layer_num}"
            # relu_layer = getattr(self, relu_layer_name)

            # 对输出值进行缩放
            layer = getattr(self, layer_name)

            def hook_fn(module, input, output, idx=idx, scaling_factor=scaling_factor):
                output[:, idx] *= (1 + scaling_factor)
                return output

            hook = layer.register_forward_hook(hook_fn)
            self.hooks.append(hook)

    def register_do_hooks(self, layer_name, neuron_idx, do_value):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

        #对激活值进行干预
        # layer_num = layer_name.lstrip('layer')
        # relu_layer_name = f"relu{layer_num}"
        # relu_layer = getattr(self, relu_layer_name)

        #对输出值进行干预
        layer = getattr(self, layer_name)

        def hook_fn(module, input, output, neuron_idx=neuron_idx, do_value=do_value):
            output[:, neuron_idx] = do_value
            return output

        # hook = relu_layer.register_forward_hook(hook_fn)
        hook = layer.register_forward_hook(hook_fn)
        self.hooks.append(hook)