package pathrag.prompt

import dev.langchain4j.model.input.PromptTemplate

/**
 * Prompt templates and symbols used throughout PathRAG.
 */
object Prompts {
    /**
     * Separator token for graph fields.
     */
    const val GRAPH_FIELD_SEP = "<SEP>"

    /**
     * Default language used in prompts.
     */
    const val DEFAULT_LANGUAGE = "English"

    /**
     * Default entity type list for extraction prompts.
     */
    val DEFAULT_ENTITY_TYPES = listOf("organization", "person", "geo", "event", "category")

    /**
     * Spinner glyphs for CLI progress updates.
     */
    val PROCESS_TICKERS = listOf("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    /**
     * Generic fallback response when an answer cannot be generated.
     */
    const val FAIL_RESPONSE = "Sorry, I'm not able to provide an answer to that question."

    /**
     * Prompt used to generate RAG answers given CSV context tables.
     */
    const val RAG_RESPONSE = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{{response_type}}

---Data tables---

{{context_data}}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

    /**
     * Prompt for keyword extraction from user queries.
     */
    const val KEYWORDS_EXTRACTION = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

---Goal---

Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Output the keywords in JSON format.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.

######################
-Examples-
######################
{{examples}}

#############################
-Real Data-
######################
Query: {{query}}
######################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

    /**
     * Examples used in keyword extraction prompts.
     */
    val KEYWORDS_EXTRACTION_EXAMPLES =
        listOf(
            """Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}
#############################""",
            """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}
#############################""",
            """Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}
#############################""",
        )

    /**
     * Prompt for summarizing entity or relationship descriptions.
     */
    const val SUMMARIZE_ENTITY_DESCRIPTIONS = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we have the full context.
Use {{language}} as output language.

#######
-Data-
Entities: {{entity_name}}
Description List: {{description_list}}
#######
Output:
"""

    /**
     * Prompt for LLM-based similarity scoring between questions.
     */
    const val SIMILARITY_CHECK = """Please analyze the similarity between these two questions:

Question 1: {{original_prompt}}
Question 2: {{cached_prompt}}

Please provide a similarity score between 0 and 1 directly:
1. Whether these two questions are semantically similar
2. Whether the answer to Question 2 can be used to answer Question 1
Return only a number between 0-1, no extra text.
"""

    /**
     * Prompt for extracting entities and relationships from text.
     */
    const val ENTITY_REL_JSON = """
-Goal-
Given a text document and a list of entity types, identify all entities of those types and all relationships among the identified entities.
Use {{language}} as output language.

-Steps-
1. Identify all entities. For each entity, extract:
   - entity_name: Name of the entity, use same language as input text. If English, capitalize the name.
   - entity_type: One of the following types: [{{entity_types}}]
   - description: Comprehensive description of the entity's attributes and activities (<= 200 chars)
2. Identify all pairs of related entities. For each relationship, extract:
   - src_id: source entity name (as identified in step 1)
   - tgt_id: target entity name (as identified in step 1)
   - description: why the source and target are related (<= 200 chars)
   - keywords: high-level keywords summarizing the relationship
   - weight: numeric strength score

-Output Format-
Return strict JSON only (no extra text) with this schema:
{
  "entities": [ { "entity_name": "...", "entity_type": "...", "description": "..." } ],
  "relationships": [ { "src_id": "...", "tgt_id": "...", "description": "...", "keywords": "...", "weight": 1 } ]
}
Use uppercase for entity_name/src_id/tgt_id values.

######################
-Examples-
######################
{{examples}}

#############################
-Real Data-
######################
Entity_types: {{entity_types}}
Text: {{text}}
######################
Output:
"""

    /**
     * JSON examples for entity/relationship extraction.
     */
    val ENTITY_EXTRACTION_EXAMPLES =
        listOf(
            """Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
Output:
{
  "entities": [
    { "entity_name": "ALEX", "entity_type": "person", "description": "A character who experiences frustration and closely observes group dynamics." },
    { "entity_name": "TAYLOR", "entity_type": "person", "description": "An authoritarian figure who shows reluctant respect toward the device." },
    { "entity_name": "JORDAN", "entity_type": "person", "description": "Shares commitment to discovery and has a significant interaction with Taylor." },
    { "entity_name": "CRUZ", "entity_type": "person", "description": "Associated with a narrowing vision of control and order." },
    { "entity_name": "THE DEVICE", "entity_type": "technology", "description": "A technology with potentially game-changing impact, revered by Taylor." }
  ],
  "relationships": [
    { "src_id": "ALEX", "tgt_id": "TAYLOR", "description": "Alex is affected by Taylor's certainty and observes shifts in attitude toward the device.", "keywords": "power dynamics, perspective shift", "weight": 7 },
    { "src_id": "ALEX", "tgt_id": "JORDAN", "description": "Alex and Jordan share a commitment to discovery that contrasts with Cruz's vision.", "keywords": "shared goals, rebellion", "weight": 6 },
    { "src_id": "TAYLOR", "tgt_id": "JORDAN", "description": "Taylor and Jordan interact directly regarding the device, leading to an uneasy truce.", "keywords": "conflict resolution, mutual respect", "weight": 8 },
    { "src_id": "JORDAN", "tgt_id": "CRUZ", "description": "Jordan's commitment to discovery is in rebellion against Cruz's vision.", "keywords": "ideological conflict, rebellion", "weight": 5 },
    { "src_id": "TAYLOR", "tgt_id": "THE DEVICE", "description": "Taylor shows reverence toward the device, indicating its importance and potential impact.", "keywords": "reverence, technological significance", "weight": 9 }
  ]
}
#############################""",
            """Example 2:

Entity_types: [person, technology, mission, organization, location]
Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols--it demanded a new perspective, a new resolve.

Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence--the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
Output:
{
  "entities": [
    { "entity_name": "WASHINGTON", "entity_type": "location", "description": "A location tied to communications influencing the team's decisions." },
    { "entity_name": "OPERATION: DULCE", "entity_type": "mission", "description": "A mission that has evolved toward interaction and preparation." },
    { "entity_name": "THE TEAM", "entity_type": "organization", "description": "A group transitioning from passive recipients to active participants." }
  ],
  "relationships": [
    { "src_id": "THE TEAM", "tgt_id": "WASHINGTON", "description": "The team receives communications from Washington that influence decisions.", "keywords": "decision-making, external influence", "weight": 7 },
    { "src_id": "THE TEAM", "tgt_id": "OPERATION: DULCE", "description": "The team is directly involved in executing Operation: Dulce.", "keywords": "mission evolution, active participation", "weight": 9 }
  ]
}
#############################""",
            """Example 3:

Entity_types: [person, role, technology, organization, event, location, concept]
Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

Alex surveyed his team--each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable--a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
Output:
{
  "entities": [
    { "entity_name": "SAM RIVERA", "entity_type": "person", "description": "A team member involved in communicating with the unknown intelligence." },
    { "entity_name": "ALEX", "entity_type": "person", "description": "The leader of a team preparing for potential first contact." },
    { "entity_name": "CONTROL", "entity_type": "concept", "description": "The ability to govern or manage, challenged by the unknown intelligence." },
    { "entity_name": "INTELLIGENCE", "entity_type": "concept", "description": "An unknown entity capable of learning and writing its own rules." },
    { "entity_name": "FIRST CONTACT", "entity_type": "event", "description": "A potential initial communication between humanity and the intelligence." },
    { "entity_name": "HUMANITY'S RESPONSE", "entity_type": "event", "description": "The collective action taken in response to the unknown message." }
  ],
  "relationships": [
    { "src_id": "SAM RIVERA", "tgt_id": "INTELLIGENCE", "description": "Sam Rivera is directly involved in learning to communicate with the intelligence.", "keywords": "communication, learning process", "weight": 9 },
    { "src_id": "ALEX", "tgt_id": "FIRST CONTACT", "description": "Alex leads the team that might be making First Contact.", "keywords": "leadership, exploration", "weight": 10 },
    { "src_id": "ALEX", "tgt_id": "HUMANITY'S RESPONSE", "description": "Alex and the team are key figures in Humanity's Response.", "keywords": "collective action, cosmic significance", "weight": 8 },
    { "src_id": "CONTROL", "tgt_id": "INTELLIGENCE", "description": "Control is challenged by an intelligence that writes its own rules.", "keywords": "power dynamics, autonomy", "weight": 7 }
  ]
}
#############################""",
        )

    /**
     * Prompt to request additional missing entities/relationships (JSON only).
     */
    const val ENTITY_CONTINUE_JSON =
        "MANY entities and relationships were missed in the last extraction. " +
            "Add them below using the same JSON schema. Respond with JSON only."

    /**
     * Prompt to decide whether further extraction loops are needed.
     */
    const val ENTITY_IF_LOOP_JSON =
        "It appears some entities/relationships may have still been missed. Answer YES | NO if there are still items that need to be added."

    /**
     * Render a prompt template using LangChain4j templating rules.
     */
    fun render(
        template: String,
        variables: Map<String, Any?>,
    ): String = PromptTemplate.from(template).apply(variables).text()
}
