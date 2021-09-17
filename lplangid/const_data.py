"""This file contains data used in classifier rules, e.g., do-not-classify system rules."""

"""Messages with these starting strings are never classified."""
COMPUTERESE_STARTS = {
    "<!", "EndToEndSystemTest", "Metadata", "[public/comment]", "displayText", "http"
}
