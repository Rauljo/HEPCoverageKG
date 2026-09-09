# Answer routing by question shape

*Status: discussed (2026-09-09). Not built.*

`--name-ids` tells the model to write the arXiv ids of every paper it asserts
into the answer text. That is the right instruction for set questions and the
wrong one for the others: on per-paper questions the answer becomes an id list
and stops describing the paper (fuzzy label recall 0.81 -> 0.67, D-153); on
counts it adds nothing; on yes/no it is noise. The generated set has the shape
on every question (`shape`: set / count / labels), and Gabriel's questions are
all sets.

Mechanism: pick the `answer` tool description by shape -- list-of-ids for
sets, "the number and the entity it was counted over" for counts, "the labels
found, in the paper's terms" for per-paper, one-line verdict for yes/no --
either from the question file's shape field (evaluation) or from a cheap
classifier on the question text (deployment). Measure per type against the
one-instruction arm (54272 / 99172), expecting the set numbers unchanged and
the per-paper and count numbers back to the control's.

Related: D-117 (the instruction), D-153 (its cost by type), D-151 (counts).
