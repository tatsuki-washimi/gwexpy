def extract_channels(products: dict) -> list[str]:
    """
    Extract unique channel names from a dictionary of products.
    Handles both single channel keys and tuple keys (chB, chA).
    """
    all_channels = set()
    for prod_name, items in products.items():
        if not isinstance(items, dict):
            continue
        for key in items.keys():
            if isinstance(key, tuple):
                for subkey in key:
                    all_channels.add(str(subkey))
            else:
                all_channels.add(str(key))
    return sorted(list(all_channels))
