import torch

# NOTE: Untested this mod.


def run_pretraining(cfg, embedding_net, thetas, xs):
    if cfg.method.pretraining.name == "infomax":
        embedding_net = pretrain_embedding_net(thetas, xs, embedding_net)
    elif cfg.method.pretraining.name == "sliced":
        embedding_net = pretrain_sliced_embedding_net(thetas, xs, embedding_net)
    else:
        raise ValueError(f"Unknown pretraining method {cfg.method.pretraining}")

    return ensure_embedding_net_freezed(embedding_net)


class FrozenModule(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        with torch.no_grad():
            return self.module(x).detach()


def ensure_embedding_net_freezed(embedding_net):
    # The prevent furthe training the output of the embedding network should stop gradients
    for param in embedding_net.parameters():
        param.requires_grad = False

    # The output of the embedding network should be detached
    embedding_net = FrozenModule(embedding_net)

    return embedding_net


def h(a, b):
    # Distance correlation loss!
    N = a.shape[0]

    term1 = torch.linalg.norm(a - b, axis=-1)
    term2 = torch.linalg.norm(a[:, None, ...] - b, axis=-1).sum(axis=0) / (N - 2)
    term3 = torch.linalg.norm(a - b[:, None, ...], axis=-1).mean(axis=0) / (N - 2)
    term4 = torch.linalg.norm(a[None, ...] - b[:, None, ...], axis=-1).sum(axis=0) / (
        (N - 1) * (N - 2)
    )
    return term1 - term2 - term3 + term4


def summary_loss_fn(summary_net, thetas, xs):
    summary_stats = summary_net(xs)

    theta = thetas[::2]
    s = summary_stats[::2]
    theta_prime = thetas[1::2]
    s_prime = summary_stats[1::2]

    h_theta = h(theta, theta_prime)
    h_s = h(s, s_prime)

    term1 = torch.sum(h_theta * h_s)
    term2 = torch.sqrt(torch.sum(h_theta**2))
    term3 = torch.sqrt(torch.sum(h_s**2))

    return -term1 / (term2 * term3)


def pretrain_embedding_net(thetas, xs, embedding_net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_net.to(device)

    optimizer = torch.optim.Adam(embedding_net.parameters(), lr=1e-3)
    val_fraction = 0.1
    num_simulations = thetas.shape[0]
    num_val = int(val_fraction * num_simulations)

    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(thetas[:-num_val], xs[:-num_val]),
        batch_size=400,
        shuffle=True,
    )

    dataloader_val = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(thetas[-num_val:], xs[-num_val:]),
        batch_size=400,
        shuffle=False,
    )

    best_loss_val = float("inf")
    patience = 50
    patience_counter = 0
    best_model_state = None
    max_num_epochs = 500

    for epoch in range(max_num_epochs):
        for thetas_batch, xs_batch in dataloader:
            thetas_batch = thetas_batch.to(device)
            xs_batch = xs_batch.to(device)

            optimizer.zero_grad()
            loss = summary_loss_fn(embedding_net, thetas_batch, xs_batch)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            loss_val = 0
            for thetas_batch, xs_batch in dataloader_val:
                thetas_batch = thetas_batch.to(device)
                xs_batch = xs_batch.to(device)
                loss_val += summary_loss_fn(embedding_net, thetas_batch, xs_batch)
            loss_val /= len(dataloader_val)

        print(f"Epoch {epoch}, loss: {loss.item()}, loss_val: {loss_val}")

        if loss_val < best_loss_val:
            best_loss_val = loss_val
            patience_counter = 0
            best_model_state = embedding_net.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping")
            break

    if best_model_state is not None:
        embedding_net.load_state_dict(best_model_state)

    embedding_net.eval()
    embedding_net.to("cpu")
    return embedding_net


def sliced_dc_loss_fn(summary_net, slice_net, thetas, xs):
    phi = torch.randn(thetas.shape)
    phi = phi / torch.linalg.norm(phi, axis=-1, keepdims=True)

    theta_sliced = torch.sum(phi * thetas, axis=-1, keepdims=True)
    summary_stats = summary_net(xs)
    summary_stats_extended = torch.concatenate(
        [summary_stats, torch.repeat_interleave(phi, 10, axis=-1)], axis=-1
    )
    summary_stats_sliced = slice_net(summary_stats_extended)

    theta = theta_sliced[::2]
    s = summary_stats_sliced[::2]
    theta_prime = theta_sliced[1::2]
    s_prime = summary_stats_sliced[1::2]

    h_theta = h(theta, theta_prime)
    h_s = h(s, s_prime)

    term1 = torch.sum(h_theta * h_s)
    term2 = torch.sqrt(torch.sum(h_theta**2))
    term3 = torch.sqrt(torch.sum(h_s**2))

    return -term1 / (term2 * term3)


def pretrain_sliced_embedding_net(thetas, xs, embedding_net):
    input_dim = embedding_net.output_dim + thetas.shape[-1] * 10
    sliced_summary_net = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 2),
    )

    # First train the embedding network!
    optimizer = torch.optim.Adam(embedding_net.parameters(), lr=1e-3)
    optimizer_slice = torch.optim.Adam(sliced_summary_net.parameters(), lr=1e-3)
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(thetas[:-1000], xs[:-1000]),
        batch_size=400,
        shuffle=True,
    )

    dataloader_val = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(thetas[-1000:], xs[-1000:]),
        batch_size=400,
        shuffle=False,
    )
    best_loss_val = float("inf")
    patience = 50
    patience_counter = 0
    best_model_state = None
    best_slice_state = None
    max_num_epochs = 500

    for epoch in range(max_num_epochs):
        for thetas_batch, xs_batch in dataloader:
            optimizer.zero_grad()
            optimizer_slice.zero_grad()
            loss = sliced_dc_loss_fn(
                embedding_net, sliced_summary_net, thetas_batch, xs_batch
            )
            loss.backward()
            optimizer.step()
            optimizer_slice.step()

        with torch.no_grad():
            loss_val = 0
            for thetas_batch, xs_batch in dataloader_val:
                loss_val += sliced_dc_loss_fn(
                    embedding_net, sliced_summary_net, thetas_batch, xs_batch
                )
            loss_val /= len(dataloader_val)

        print(f"Epoch {epoch}, loss: {loss.item()}, loss_val: {loss_val}")

        if loss_val < best_loss_val:
            best_loss_val = loss_val
            patience_counter = 0
            best_model_state = embedding_net.state_dict().copy()
            best_slice_state = sliced_summary_net.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping")
            break

    if best_model_state is not None:
        embedding_net.load_state_dict(best_model_state)
        sliced_summary_net.load_state_dict(best_slice_state)

    embedding_net.eval()
    sliced_summary_net.eval()

    return embedding_net
