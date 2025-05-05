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
