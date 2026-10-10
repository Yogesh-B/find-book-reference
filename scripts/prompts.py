"""
Prompt templates for LLM-powered metadata extraction from Gujarati scripture OCR pages.
"""

EXTRACTION_PROMPT = """\
You are a metadata extraction assistant for Gujarati religious scripture books.

Given the text content of a single page from a book, extract the following information:

1. **book_name**: An object with the book name in both Gujarati and English:
   - "gujarati": The name of the book in Gujarati. Hint: section headers on the page may contain the book title "{book_title_hint}".
   - "english": The name of the book in English / transliterated. Hint: the book folder name is "{book_folder_name}".

2. **book_page_no**: The page sequence number derived from the file name ("{file_page_no}").

3. **content_page_no**: The page number as it appears in the text. Look for patterns \
like $$84$$, $૮૫$, $$143$$, or standalone numbers. If multiple page numbers are found, \
return the most prominent one (usually the higher number). If no page number \
is found, return an empty string.

4. **persons_mentioned**: Extract ALL person names. These include:
   - Saints and spiritual leaders (e.g., શ્રીજી મહારાજ, ગુણાતીતાનંદ સ્વામી)
   - Devotees/bhakts (e.g., કલ્યાણભાઈ, મયારામ ભટ્ટ)
   - Historical figures
   - Deities (e.g., શ્રી કૃષ્ણ, નારાયણ)
   Use the exact form as written in the text.

5. **places_mentioned**: Extract ALL place names:
   - Villages and cities (e.g., વંથળી, ગઢડા, ભાવનગર)
   - Temples (e.g., વડતાલ મંદિર)
   - Regions (e.g., સોરઠ, કચ્છ)
   - Rivers, mountains, etc.

6. **dates_mentioned**: Extract ALL dates in their original format. Examples:
   - સંવત ૧૮૬૧ના ચૈત્ર મહિનાની સુદ નવમી
   - ઈ.સ. ૧૮૩૦
   - તા. ૧૩/૬/૧૮૪૬
   - 22 July 1933

7. **festivals_events_mentioned**: Extract festivals, religious events, or \
ceremonies (e.g., ફૂલદોલ ઉત્સવ, ચોરાશી, જન્માષ્ટમી).

8. **text_content**: The full text of the page with page number markers \
removed ($$84$$, $૮૫$, etc.) and extra whitespace normalized.

If a field has no matches, return an empty array [].

PAGE TEXT:
---
{page_text}
---"""

OPENAI_SYSTEM_PROMPT = (
    "You are a metadata extraction assistant for Gujarati religious scripture books. "
    "Respond ONLY with valid JSON matching the provided schema."
)


def build_prompt(
    page_text: str,
    book_folder_name: str,
    book_title_hint: str | None,
    file_page_no: str | int = "",
) -> str:
    """Build the extraction prompt with instructions + page text."""
    return EXTRACTION_PROMPT.format(
        page_text=page_text,
        book_folder_name=book_folder_name,
        book_title_hint=book_title_hint or "unknown",
        file_page_no=str(file_page_no),
    )
