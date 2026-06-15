"""Stateless survival label/bin/target transforms (no LightningModule state)."""
import torch


def survival_year_values(values, month_values=None, default=None):
    if values is None and month_values is not None:
        values = [float(value) / 12.0 for value in month_values]
    if values is None:
        values = default or []
    return [float(value) for value in values]


def format_survival_landmark_label(value):
    value = float(value)
    if value.is_integer():
        return f"{int(value)}y"
    return f"{value:g}y".replace(".", "p")


def time_to_survival_bin(continuous_time, cut_points_years, num_time_bins):
    cut_points = cut_points_years.to(continuous_time.device)
    time_bin = torch.bucketize(
        continuous_time.float(),
        cut_points,
        right=False,
    )
    return time_bin.clamp(0, num_time_bins - 1).long()


def interval_frac(continuous_time, time_bin, bin_edges):
    """Fractional position of continuous_time within its bin, in [0, 1].

    Left edges are 0, cut_points[0], cut_points[1], ..., cut_points[K-2].
    For non-uniform bins each bin's width equals edges[k+1] - edges[k];
    the last bin's width equals that of the second-to-last bin (extrapolated).
    """
    edges = bin_edges.to(continuous_time.device)
    tb = time_bin.clamp(0, len(edges) - 1)
    left = edges[tb]
    # Per-bin width: edges[k+1] - edges[k]. For the last bin use the
    # previous interval's width (edges has length K = num_time_bins).
    next_idx = (tb + 1).clamp(0, len(edges) - 1)
    width = edges[next_idx] - edges[tb]
    # When tb == last bin, next_idx == tb so width == 0; fall back to
    # the second-to-last interval width to avoid division by zero.
    if len(edges) > 1:
        last_valid_width = (edges[-1] - edges[-2]).clamp_min(1e-7)
    else:
        last_valid_width = torch.tensor(1.0, device=edges.device)
    width = torch.where(width > 0, width, last_valid_width)
    return ((continuous_time.float() - left) / width).clamp(0.0, 1.0)


def unpack_survival_targets(y, device, cut_points_years, num_time_bins):
    if isinstance(y, dict):
        event = y["event"]
        if "time" in y:
            continuous_time = y["time"]
            time_bin = None
        elif "time_years" in y:
            continuous_time = y["time_years"]
            time_bin = None
        elif "time_months" in y:
            continuous_time = y["time_months"].float() / 12.0
            time_bin = None
        elif "time_bin" in y:
            continuous_time = y["time_bin"]
            time_bin = y["time_bin"].to(
                device=device,
                dtype=torch.long,
            ).view(-1)
        else:
            raise KeyError(
                "Survival targets must contain 'time', 'time_years', "
                "'time_months', or legacy 'time_bin'."
            )
    else:
        time_bin, event = y
        continuous_time = time_bin
        time_bin = time_bin.to(device=device, dtype=torch.long).view(-1)

    continuous_time = continuous_time.to(
        device=device,
        dtype=torch.float32,
    ).view(-1)
    if time_bin is None:
        time_bin = time_to_survival_bin(continuous_time, cut_points_years, num_time_bins)

    return (
        time_bin,
        event.to(device=device, dtype=torch.float32).view(-1),
        continuous_time,
    )


def survival_label_tensor(time_bin, event):
    return torch.stack([time_bin.float(), event.float()], dim=1)
