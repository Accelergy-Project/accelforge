"""
These classes may be useful when constraining joining pmappings. For example,
if pmapping A is compatible with any other with the same dataflow (say,
dataflow DA), and pmapping B is conly compatible with pmappings with dataflow
DA and untiled weights in the global buffer, we can tag pmapping A with {DA}
and pmapping B with {DA, untiled_weights}.

It is often useful to compare tags that are "compatible" and tags that "match".
Moreover, we may want to use either "compatibility" or "matching" as comparison
for creating a dictionary. For convenience, one can use TagMatch and
TagCompatibility as classes for the keys.
"""

from fastfusion.util import fzs


class TagClass(fzs):
    pass


class Tags(fzs):
    def __repr__(self):
        return f"Tags(({super().__repr__()}))"

    def __str__(self):
        return f"Tags({super().__repr__()})"

    def is_member_of(self, tag_class: TagClass):
        return all(class_string in self for class_string in tag_class)

    def are_compatible_with(self, tag2):
        return all(tag2_string in self for tag2_string in tag2) or all(
            tag1_string in tag2 for tag1_string in self
        )

    # def filter_membership(tags: set["Tag"], tag_class: TagClass) -> set["Tag"]:
    #     return {tag for tag in tags if are_compatible_with(tag, tag_class)}

    def matches(self, tag2):
        return self == tag2

    @staticmethod
    def from_tuple(t: tuple):
        return Tags(t)


class TagMatch:
    def __init__(self, tags: Tags):
        self.tags = tags

    def __str__(self):
        return f"TagMatch({repr(self.tags)})"

    def __hash__(self):
        return hash(self.tags)

    def __eq__(self, other: "TagMatch"):
        return self.tags.matches(other.tags)


class TagCompatibility:
    def __init__(self, tags: Tags):
        self.tags = tags

    def __hash__(self):
        return 0  # See note below

    def __eq__(self, other: "TagCompatibility"):
        return self.tags.are_compatible_with(other.tag)
