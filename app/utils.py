def build_legend_str(counts_dict, class_names):
    return " | ".join(
        [f"{class_names[i]}: {counts_dict[i]}" for i in range(len(class_names))]
    )
