"""
Read the header of one Arrow IPC stream message, without its body (#419).

`Salesforce/GiftEvalPretrain` stores each source as Arrow IPC *stream* files
(`datasets.save_to_disk`). A stream file has no footer, so the only way to
find its record batches is to walk the messages from the start. Each message
is a continuation marker, a flatbuffer header and a body. The header gives
the body length, so a walk can skip every body and read a few kB per record
batch. `scripts/build_gift_pretrain_index.py` does that walk once.

pyarrow reads a whole message (header and body) and cannot skip the body, so
this module reads the flatbuffer fields it needs by hand. The layout is the
one of `format/Message.fbs` in the Arrow repository.
"""

import struct

CONTINUATION = 0xFFFFFFFF
SCHEMA = 1
RECORD_BATCH = 3


def _vtable(buf, table):
    """Field offsets of the flatbuffer table at ``table`` (0 = absent)."""
    vt = table - struct.unpack_from("<i", buf, table)[0]
    n = (struct.unpack_from("<H", buf, vt)[0] - 4) // 2
    return [struct.unpack_from("<H", buf, vt + 4 + 2 * i)[0] for i in range(n)]


def _scalar(buf, table, fields, i, fmt):
    """Scalar field ``i`` of a table, or 0 when the field is absent."""
    if i >= len(fields) or fields[i] == 0:
        return 0
    return struct.unpack_from(fmt, buf, table + fields[i])[0]


def _ref(buf, table, fields, i):
    """Position of the table or vector that field ``i`` points to."""
    if i >= len(fields) or fields[i] == 0:
        return None
    at = table + fields[i]
    return at + struct.unpack_from("<I", buf, at)[0]


def _pairs(buf, vec):
    """A vector of two-int64 structs (FieldNode or Buffer)."""
    if vec is None:
        return []
    n = struct.unpack_from("<I", buf, vec)[0]
    return [struct.unpack_from("<qq", buf, vec + 4 + 16 * i) for i in range(n)]


def record_batch_fields(buf, table):
    """Rows, nodes (length, null count) and buffers (offset, length) of a
    RecordBatch table, and whether its body is compressed."""
    f = _vtable(buf, table)
    return {"rows": _scalar(buf, table, f, 0, "<q"),
            "nodes": _pairs(buf, _ref(buf, table, f, 1)),
            "buffers": _pairs(buf, _ref(buf, table, f, 2)),
            "compressed": _ref(buf, table, f, 3) is not None}


def message_header(meta):
    """Parse a message's flatbuffer header: its type, its body length and,
    for a record batch, :func:`record_batch_fields`."""
    root = struct.unpack_from("<I", meta, 0)[0]
    f = _vtable(meta, root)
    kind = _scalar(meta, root, f, 1, "<B")
    out = {"type": kind, "body_length": _scalar(meta, root, f, 3, "<q")}
    if kind == RECORD_BATCH:
        out.update(record_batch_fields(meta, _ref(meta, root, f, 2)))
    return out


def split_prefix(head):
    """``(metadata length, prefix length)`` of a message starting at byte 0
    of ``head``. The prefix is the continuation marker and the length."""
    marker, length = struct.unpack_from("<Ii", head, 0)
    if marker != CONTINUATION:
        raise ValueError(f"not an Arrow IPC stream message (marker {marker:#x})")
    return length, 8


# ── Column layout ────────────────────────────────────────────────────────────


def type_layout(t):
    """``(nodes, buffers)`` an Arrow type takes in a record batch, depth
    first: a list adds a validity and an offsets buffer, a fixed-size list
    a validity buffer, a string three buffers, a primitive two."""
    import pyarrow as pa
    if pa.types.is_fixed_size_list(t):
        n, b = type_layout(t.value_type)
        return 1 + n, 1 + b
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        n, b = type_layout(t.value_type)
        return 1 + n, 2 + b
    if pa.types.is_string(t) or pa.types.is_binary(t):
        return 1, 3
    return 1, 2


def field_starts(schema):
    """First node and first buffer of every field, by name."""
    starts, node, buf = {}, 0, 0
    for field in schema:
        starts[field.name] = (node, buf)
        n, b = type_layout(field.type)
        node, buf = node + n, buf + b
    return starts


def float_column(schema, name):
    """Where a list-of-floats column lives in a record batch.

    Returns ``(variates per row, node of the floats, buffer of the offsets,
    buffer of the float validity)``. A ``list<float>`` column has 1 variate
    per row; a ``fixed_size_list<list<float>>[D]`` column has D, and its
    offsets are those of the inner list. The float values follow the
    validity buffer.
    """
    import pyarrow as pa
    node, buf = field_starts(schema)[name]
    t = schema.field(name).type
    if pa.types.is_fixed_size_list(t):
        return t.list_size, node + 2, buf + 2, buf + 3
    return 1, node + 1, buf + 1, buf + 2
