from scripts.update_arxiv_data import clean_latex_artifacts


def test_unwraps_text_formatting_commands():
    assert clean_latex_artifacts(r"This is \textbf{robust} and \emph{fast}.") == (
        "This is robust and fast."
    )


def test_unescapes_special_characters():
    assert clean_latex_artifacts(r"Accuracy improves by 5\% with \&-rule features.") == (
        "Accuracy improves by 5% with &-rule features."
    )


def test_replaces_line_break_command_with_space():
    assert clean_latex_artifacts("First line.\\\\Second line.") == "First line. Second line."


def test_strips_bare_math_dollar_delimiters():
    assert clean_latex_artifacts(r"The loss is $O(n^2)$ in the worst case.") == (
        "The loss is O(n^2) in the worst case."
    )


def test_passes_through_plain_text_unchanged():
    assert clean_latex_artifacts("A plain abstract with no LaTeX.") == (
        "A plain abstract with no LaTeX."
    )


def test_handles_empty_and_none_input():
    assert clean_latex_artifacts("") == ""
    assert clean_latex_artifacts(None) == ""
