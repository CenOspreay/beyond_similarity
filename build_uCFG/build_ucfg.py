from build_uCFG.build_uir import build_uir


def build_ucfg_from_uir(uirs):
    id2u = {u["id"]: u for u in uirs}
    edges = []

    def add(u, v, t):
        if u is not None and v is not None:
            edges.append((u, v, t))

    # 找 loop 体的最后一句
    def get_loop_last(loop_id):
        body_nodes = [u for u in uirs if u["NLS"] == loop_id]
        if not body_nodes:
            return None
        return max(body_nodes, key=lambda x: x["id"])["id"]

    for u in uirs:
        uid = u["id"]
        cat = u["category"]
        NS  = u["NS"]
        NSS = u["NSS"]
        NLS = u["NLS"]

        # --- Subroutine ---
        if cat == "Subroutine":
            add(uid, NSS, "Next")
            continue

        # --- Loop ---
        if cat == "Loop":
            # loop → body
            add(uid, NSS, "Loop")

            # body_last → loop_head
            last_body = get_loop_last(uid)
            if last_body:
                add(last_body, uid, "LoopBack")

            # loop_exit
            add(uid, NS, "Exit")
            continue

        # --- Conditional ---
        if cat == "Conditional":
            # true
            add(uid, NSS, "True")
            # false
            add(uid, NLS, "False")
            continue

        # --- Statement ---
        if cat == "Statement":
            # normally → loop head
            if NLS is not None:
                add(uid, NLS, "LoopBack")
            continue

        # --- Return ---
        if cat == "Return":
            # no edge needed (function end)
            continue

    nodes = [{"id": u["id"], "label": u["label"], "category": u["category"]} for u in uirs]

    return {"nodes": nodes, "edges": edges}

