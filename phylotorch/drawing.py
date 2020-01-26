import matplotlib.pyplot as plt


def draw_tree(fig, tree, **kwargs):
    ax = fig.gca()

    fig_width = fig.get_dpi()

    yspace = kwargs.get('yspace', 10)
    left_offset = kwargs.get('leftOffset', 5)
    branch_width = kwargs.get('branchWidth', 2)
    taxon_font_size = kwargs.get('taxonFontSize', 12)
    branch_tip_space = kwargs.get('branchTipSpace', 1.0)
    root_length = kwargs.get('rootLength', 0.0)

    showHeights = kwargs.get('showHeights', True)
    showTaxa = kwargs.get('showTaxa', True)
    showIntervals = kwargs.get('showIntervals', True)
    showHeightLabels = kwargs.get('showHeightLabels', False)

    r = fig.canvas.get_renderer()
    scaler = float("inf")

    # keep track of coordinates for each node. Usefull for drawing the vertical branches
    coordinates = {}

    # contains a cleanup taxon name and its dimension in the plot
    taxa_labels = {}

    taxaCount = 0
    taxon_height = 0
    for node in tree:
        if len(node.children) == 0:
            taxaCount += 1

            if showTaxa:
                t = plt.text(-1, -1, node.name, alpha=0, size=taxon_font_size)
                bb = t.get_window_extent(renderer=r)
                taxa_labels[node] = (node.name, bb.width, bb.height)
                taxon_height = bb.height
                s = (fig_width - branch_tip_space - bb.width) / (tree.height - node.height)
                scaler = min(s, scaler)

    if showTaxa == False:
        scaler = fig_width / tree.height

    y = taxon_height

    yy = taxon_height + taxaCount * (yspace + taxon_height)

    intervals = []

    for node in tree:
        x1 = ((tree.height - node.height) * scaler) + left_offset + root_length

        if node.parent is not None:
            x2 = x1 - (node.parent.height - node.height) * scaler

        if len(node.children) == 0:
            y1 = y2 = y - taxon_height * 0.5
            if showTaxa:
                ax.text(x1 + branch_tip_space, y - (taxon_height - branch_width) * 0.5, taxa_labels[node][0], va='top',
                        size=taxon_font_size)
            y += yspace + taxon_height
        else:
            y1, y2 = [coordinates[c]['endY'] for c in node.children]
            # Vertical branch
            ax.plot([x1, x1], [y1, y2], lw=branch_width, color='steelblue', ls='-', zorder=9)

            y1 = y2 = y1 + (y2 - y1) / 2.0  # midpoint between childs

            if showHeights:
                ax.text(x1 + 1.0, y1, "%.3f" % (node.height), size=taxon_font_size)

        # Horizontal branch
        if node.parent is not None:
            node_branch_length = node.parent.height - node.height
            if node_branch_length is not None and node_branch_length > 0.0:
                ax.plot([x1, x2], [y1, y2], lw=branch_width, color='steelblue', ls='-', zorder=9)
                coordinates[node] = {'startX': x2, 'endX': x1, 'startY': y2, 'endY': y2}
            elif node_branch_length is None and root_length > 0.0:
                ax.plot([x1, x1 - root_length], [y1, y2], lw=branch_width, color='steelblue', ls='-', zorder=9)

    if showIntervals:
        nodes = sorted(coordinates.keys(), key=lambda n: n.height, reverse=True)

        x2 = left_offset + root_length
        ax.plot([x2, x2], [0, yy], '--', c='black')

        for i, node in enumerate(nodes):
            x1 = coordinates[node]['endX']
            ax.plot([x1, x1], [0, yy], '--', c='black')
            ax.text((x1 - x2) / 2 + x2, 0, r'$u_{}$'.format(i + 2), size=16)
            x2 = x1
            if node.height < 1.0e-10:
                break

    if showHeightLabels:
        nodes = sorted(coordinates.keys(), key=lambda n: n.height, reverse=True)

        x2 = left_offset + root_length
        ax.plot([x2, x2], [0, yy], '--', c='black')

        for i, node in enumerate(nodes):
            x1 = coordinates[node]['endX']
            ax.plot([x1, x1], [0, yy], '--', c='black')
            ax.text(x2, yy, r'$t_{}$'.format(i + 1), size=16)
            x2 = x1
            if node.height < 1.0e-10:
                break

        ax.text(x1, yy, r'$t_{}$'.format(i + 2), size=16)

    ax.axis('off')
