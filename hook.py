import torch.nn as nn
import torch


class ItemIterator:
    @property
    def iterator_item(self):
        raise NotImplementedError

    def __iter__(self):
        return iter(self.iterator_item)

    def __getitem__(self, item):
        return self.iterator_item[item]

    def __len__(self):
        return len(self.iterator_item)


class BasicHook:
    def __init__(self, module: nn.Module):
        self.hook = module.register_forward_hook(self.base_hook_fn)
        self.activations = None

    def close(self):
        self.hook.remove()

    def base_hook_fn(self, model: nn.Module, input_t: torch.tensor, _: torch.tensor):
        x = input_t
        x = x[0][0] if isinstance(x[0], tuple) else x[0]
        return self.hook_fn(model, x)

    def hook_fn(self, model: nn.Module, x: torch.tensor):
        raise NotImplementedError


class FeatureKeeper(BasicHook, ItemIterator):
    def __init__(self, module: nn.Module):
        super().__init__(module)

    @property
    def iterator_item(self):
        return self.activations

    def base_hook_fn(self, model: nn.Module, _: torch.tensor, output: torch.tensor):
        return self.hook_fn(model, output)

    def hook_fn(self, model: nn.Module, input_t: torch.Tensor):
        if self.activations is None:
            self.reset()
        self.activations.append(input_t[:, :, :-1])

    def reset(self):
        self.activations = []


class HookHolder(ItemIterator):
    def __init__(self, classifier: nn.Module, hook_class, layer_class):
        self.hooks = [hook_class(m) for m in classifier.modules() if isinstance(m, layer_class)]

    @property
    def iterator_item(self):
        return self.hooks

    def check_for_attr(self, attr: str, hook_class):
        for h in self:
            if not hasattr(h, attr):
                raise AttributeError('Class {} does not have attribute {}'.format(hook_class.__name__, attr))

    def _broadcast(self, func_name: str, *to_propagate):
        for i in self:
            func = getattr(i, func_name)
            func(*to_propagate)

    def _gather(self, attr: str) -> list:
        return [getattr(l, attr) for l in self]

    def close(self):
        self._broadcast('close')


class ActivationHooks(HookHolder):
    def __init__(self, classifier: nn.Module, hook_class, layer_class):
        super().__init__(classifier, hook_class, layer_class)
        self.check_for_attr('activations', hook_class)

    def reset(self):
        self._broadcast('reset')

    def __call__(self) -> []:
        return self[0].activations


class ModelWithHook:
    def __init__(self, model: nn.Module, layer_class, layer: int, feature: int):
        self.model = model
        self.hook = ActivationHooks(self.model, FeatureKeeper, layer_class)
        self.l = layer
        self.f = feature

    def __call__(self, wav: torch.tensor, size: torch.tensor) -> torch.tensor:
        self.model(wav, size)
        return self.hook()[self.l][0][self.f]
