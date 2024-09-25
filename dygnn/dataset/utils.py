"""Utility functions for dataset module."""


def extract_author_fields(authors_list) -> tuple[list[str], list[str], list[str]]:
    """
    Extract author fields from a list of authors.

    Args:
        authors_list: List of authors.

    Returns:
        author_ids: List of author IDs.
        author_names: List of author names.
        author_orgs: List of author organizations.
    """
    if not authors_list:
        return None, None, None
    author_ids = []
    author_names = []
    author_orgs = []
    for author in authors_list:
        author_ids.append(author.get("id", None))
        author_names.append(author.get("name", None))
        author_orgs.append(author.get("org", None))
    return author_ids, author_names, author_orgs
