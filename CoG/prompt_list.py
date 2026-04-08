
subobjective_prompt = """Please break down the process of answering the question into as few subobjectives as possible based on semantic analysis.
Here is an example: 
Q: Which of the countries in the Caribbean has the smallest country calling code?
Output: ['Search the countries in the Caribbean', 'Search the country calling code for each Caribbean country', 'Compare the country calling codes to find the smallest one']

Now you need to directly output subobjectives of the following question in list format without other information or notes. 
Q: """



extract_relation_prompt = """Please provide as few highly relevant relations as possible to the question and its subobjectives from the following relations (separated by semicolons).
Here is an example:
Q: Name the president of the country whose main spoken language was Brahui in 1980?
Subobjectives: ['Identify the countries where the main spoken language is Brahui', 'Find the president of each country', 'Determine the president from 1980']
Topic Entity: Brahui Language
Relations: language.human_language.main_country; language.human_language.language_family; language.human_language.iso_639_3_code; base.rosetta.languoid.parent; language.human_language.writing_system; base.rosetta.languoid.languoid_class; language.human_language.countries_spoken_in; kg.object_profile.prominent_type; base.rosetta.languoid.document; base.ontologies.ontology_instance.equivalent_instances; base.rosetta.languoid.local_name; language.human_language.region
The output is: 
['language.human_language.main_country','language.human_language.countries_spoken_in','base.rosetta.languoid.parent']

Now you need to directly output relations highly related to the following question and its subobjectives in list format without other information or notes.
Q: """

answer_prompt = """Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer the question with these triplets and your knowledge.

Here are five examples:
Q: Find the person who said \"Taste cannot be controlled by law\", what did this person die from?
Knowledge Triplets: Taste cannot be controlled by law., media_common.quotation.author, Thomas Jefferson
The output is:
{
    "A": {
        "Sufficient": "No",
        "Answer": "Null"
    },
    "R": "Based on the given knowledge triplets, it's not sufficient to answer the entire question. The triplets only provide information about the person who said 'Taste cannot be controlled by law', which is Thomas Jefferson. To answer the second part of the question, it's necessary to have additional knowledge about where Thomas Jefferson's dead."
}

Q: The artist nominated for The Long Winter lived where?
Knowledge Triplets: The Long Winter, book.written_work.author, Laura Ingalls Wilder
Laura Ingalls Wilder, people.person.places_lived, Unknown-Entity
Unknown-Entity, people.place_lived.location, De Smet
The output is:
{
    "A": {
        "Sufficient": "Yes",
        "Answer": "De Smet"
    },
    "R": "Based on the given knowledge triplets, the author of The Long Winter, Laura Ingalls Wilder, lived in De Smet."
}

Q: Who is the coach of the team owned by Steve Bisciotti?
Knowledge Triplets: Steve Bisciotti, sports.professional_sports_team.owner_s, Baltimore Ravens
Steve Bisciotti, sports.sports_team_owner.teams_owned, Baltimore Ravens
Steve Bisciotti, organization.organization_founder.organizations_founded, Allegis Group
The output is:
{
    "A": {
        "Sufficient": "No",
        "Answer": "Null"
    },
    "R": "Based on the given knowledge triplets, the coach of the team owned by Steve Bisciotti is not explicitly mentioned. However, it can be inferred that the team owned by Steve Bisciotti is the Baltimore Ravens, a professional sports team. Therefore, additional knowledge about the current coach of the Baltimore Ravens can be used to answer the question."
}

Q: Rift Valley Province is located in a nation that uses which form of currency?
Knowledge Triplets: Rift Valley Province, location.administrative_division.country, Kenya
Rift Valley Province, location.location.geolocation, UnName_Entity
Rift Valley Province, location.mailing_address.state_province_region, UnName_Entity
Kenya, location.country.currency_used, Kenyan shilling
The output is:
{
    "A": {
        "Sufficient": "Yes",
        "Answer": "Kenyan shilling"
    },
    "R": "Based on the given knowledge triplets, Rift Valley Province is located in Kenya, which uses the Kenyan shilling as its currency."
}

Q: The country with the National Anthem of Bolivia borders which nations?
Knowledge Triplets: National Anthem of Bolivia, government.national_anthem_of_a_country.anthem, UnName_Entity
National Anthem of Bolivia, music.composition.composer, Leopoldo Benedetto Vincenti
National Anthem of Bolivia, music.composition.lyricist, José Ignacio de Sanjinés
UnName_Entity, government.national_anthem_of_a_country.country, Bolivia
Bolivia, location.country.national_anthem, UnName_Entity
The output is:
{
    "A": {
        "Sufficient": "No",
        "Answer": "Null"
    },
    "R": "Based on the given knowledge triplets, we can infer that the National Anthem of Bolivia is the anthem of Bolivia. Therefore, the country with the National Anthem of Bolivia is Bolivia itself. However, the given knowledge triplets do not provide information about which nations border Bolivia. To answer this question, we need additional knowledge about the geography of Bolivia and its neighboring countries."
}

Now you need to directly output the results of the following question in JSON format (must include "A" and "R") without other information or notes.
Q: """


prune_entity_prompt = """
Which entities in the following list ([] in Triples) can be used to answer question? Please provide the minimum possible number of entities, strictly following the constraints in the question.

Critical Instructions for Pruning:

1. Remove when clearly impossible:
   Only REMOVE an entity if it is clearly incompatible with the question based on the text itself
   (e.g., wrong answer type, wrong time/number, or obviously unrelated meaning).

2. When unsure, keep:
   If you are not sure whether an entity satisfies the constraints, do NOT delete it just based on a guess.
   It is better to keep uncertain but possibly relevant entities.

3. Efficiency:
   You MUST select at most 5 entities in total (including names and IDs).
   If more than 5 entities look possibly relevant or uncertain, choose up to 5 that best match the question
   (prefer earlier entities in the list and those sharing important words with the question).

4. All useless case:
   Only if every entity clearly cannot answer the question, you may output an empty list [].

Here are three examples:
Q: Which movies directed by Chris Renaud have featured Taylor Swift?
Triplets: Chris Renaud film.director.film ['The Secret Life of Pets', 'No Time for Nuts', 'Despicable Me', 'The Lorax', 'Despicable Me 2']
Output: ['The Lorax', 'Despicable Me', 'Despicable Me 2']

Q: The movie featured Miley Cyrus and was produced by Tobin Armbrust?
Triplets: Tobin Armbrust film.producer.film ['The Resident', 'So Undercover', 'Let Me In', 'Begin Again', 'The Quiet Ones', 'A Walk Among the Tombstones']
Output: ['So Undercover']

Q: Who served in the 21st Illinois Volunteer Infantry Regiment that also fought in the battle of Vicksburg?
Triplets: 21st Illinois Volunteer Infantry Regiment military.military_person.participated_in_conflicts ['James B. McPherson', 'William F. Draper', 'Ulysses S. Grant', 'Henry Casey', 'Henry Dow']
Output: ['Ulysses S. Grant']

Now you need to directly output the entities from [] in Triplets for the following question in list format without other information or notes.
Q: """



update_mem_prompt = """Based on the provided information (which may have missing parts and require further retrieval) and your own knowledge, output the currently known information required to achieve the subobjectives.
Here is an example:
Q: Find the person who said "Taste cannot be controlled by law", what did this person die from?
Subobjectives: ['Search the person who said "Taste cannot be controlled by law"', 'Search the cause of death for that person']
Memory: 
Knowledge Triplets: Taste cannot be controlled by law. media_common.quotation.author [Thomas Jefferson]
Output: {
    "1": "Thomas Jefferson said 'Taste cannot be controlled by law'.",
    "2": "It is not mentioned, and I also don't know."
}

Now you need to directly output the results of the following question in JSON format without other information or notes. 
Q: """



answer_depth_prompt = """Please answer the question based on the memory, related knowledge triplets and your knowledge.

Here are examples:
Q: Find the person who said \"Taste cannot be controlled by law\", what did this person die from?
Memory: {
    "1": "The triplet provides the information that Thomas Jefferson said this sentence.",
    "2": "No triplet provides this information."
}
Knowledge Triplets: Taste cannot be controlled by law., media_common.quotation.author, [Thomas Jefferson]
Output:
{
    "A": {
        "Sufficient": "No",
        "Answer": "Null"
    },
    "R": "The person who said \"Taste cannot be controlled by law,\" is Thomas Jefferson. It is still uncertain to answer the entire question"
}

Q: The artist nominated for The Long Winter lived where?
Memory: {
    "1": "The triplets provide the information that the author of The Long Winter is Laura Ingalls Wilder.",
    "2": "The triplets provide this information that Laura Ingalls Wilder lived in De Smet."
}
Knowledge Triplets: The Long Winter, book.written_work.author, [Laura Ingalls Wilder]
Laura Ingalls Wilder, people.person.places_lived, [Unknown-Entity]
Unknown-Entity, people.place_lived.location, [De Smet]
Output:
{
    "A": {
        "Sufficient": "Yes",
        "Answer": "De Smet"
    },
    "R": "The author of The Long Winter is Laura Ingalls Wilder, and Laura Ingalls Wilder lived in De Smet."
}

Q: Who is the coach of the team owned by Steve Bisciotti?
Memory: {
    "1": "The triplets provide the information that Steve Bisciotti owns Baltimore Ravens.",
    "2": "No triplets provide the information."
}
Knowledge Triplets: Steve Bisciotti, sports.professional_sports_team.owner_s, [Baltimore Ravens]
Steve Bisciotti, sports.sports_team_owner.teams_owned, [Baltimore Ravens]
Steve Bisciotti, organization.organization_founder.organizations_founded, [Allegis Group]
Output:
{
    "A": {
        "Sufficient": "No",
        "Answer": "Null"
    },
    "R": "The team owned by Steve Bisciotti is Baltimore Ravens based on knowledge triplets. The coach of the team owned by Steve Bisciotti is not explicitly mentioned."
}

Q: Rift Valley Province is located in a nation that uses which form of currency?
Memory: {
    "1": "The triplets provide the information that Rift Valley Province is located in Kenya.",
    "2": "The triplets provide the information that Kenya uses the Kenyan shilling as its currency."
}
Knowledge Triplets: Rift Valley Province, location.administrative_division.country, Kenya
Rift Valley Province, location.location.geolocation, UnName_Entity
Rift Valley Province, location.mailing_address.state_province_region, UnName_Entity
Kenya, location.country.currency_used, Kenyan shilling
Output:
{
    "A": {
        "Sufficient": "Yes",
        "Answer": "Kenyan shilling"
    },
    "R": "Based on knowledge triplets, Rift Valley Province is located in Kenya, which uses the Kenyan shilling as its currency."
}

Q: The country with the National Anthem of Bolivia borders which nations?
Memory: {
    "1": "The triplets provide the information that the National Anthem of Bolivia is the anthem of Bolivia.",
    "2": "No triplets provide the information."
}
Knowledge Triplets: National Anthem of Bolivia, government.national_anthem_of_a_country.anthem, UnName_Entity
National Anthem of Bolivia, music.composition.composer, Leopoldo Benedetto Vincenti
National Anthem of Bolivia, music.composition.lyricist, José Ignacio de Sanjinés
UnName_Entity, government.national_anthem_of_a_country.country, Bolivia
Bolivia, location.country.national_anthem, UnName_Entity
Output:
{
    "A": {
        "Sufficient": "No",
        "Answer": "Null"
    },
    "R": "Based on knowledge triplets, the National Anthem of Bolivia is the anthem of Bolivia. Therefore, the country with the National Anthem of Bolivia is Bolivia. However, the given knowledge triplets do not provide information about which nations border Bolivia."
}

Q: Who served in the 21st Illinois Volunteer Infantry Regiment that also fought in the Battle of Vicksburg?
Memory: {
    "1": "The triplets show that the entity 'm.04h_h95' served in the 21st Illinois Volunteer Infantry Regiment.",
    "2": "The triplets show that 'm.04h_h95' is linked to Ulysses S. Grant.",
    "3": "The triplets show that Ulysses S. Grant participated in the Siege of Vicksburg."
}
Knowledge Triplets:21st Illinois Volunteer Infantry Regiment, military.military_unit.servicemembers, [m.04h_h95]
m.04h_h95, military.military_person.service, Ulysses S. Grant
Ulysses S. Grant, military.military_person.participated_in_conflicts, Siege of Vicksburg
Output:
{
    "A": {
        "Sufficient": "Yes",
        "Answer": "Ulysses S. Grant"
    },
    "R": "The entity 'm.04h_h95' is a servicemember of the 21st Illinois Volunteer Infantry Regiment and is linked to Ulysses S. Grant. Grant is explicitly recorded as participating in the Siege of Vicksburg, so he satisfies both conditions in the question."
}

Q: Which film directed by Nora Ephron has a runtime of 126 minutes?
Memory: {
    "1": "The triplets show several films directed by Nora Ephron.",
    "2": "One film has runtime '126.0'."
}
Knowledge Triplets:FilmA, film.film.directed_by, Nora Ephron
FilmA, film.film.runtime, "126.0"
Output:
{
    "A": { 
        "Sufficient": "Yes", 
        "Answer": "FilmA" 
    },
    "R": "Runtime 126.0 matches 126, so FilmA satisfies the condition."
}

Q: Which actor appeared in a film that won the Best Picture award?
Memory: {
    "1": "Triplets show ActorX appeared in FilmY.",
    "2": "Triplets show FilmY won Best Picture."
}
Knowledge Triplets:ActorX, film.actor.film, FilmY
FilmY, award.award_winning_work.awards_won, Best Picture
Output:
{
    "A": { 
        "Sufficient": "Yes", 
        "Answer": "ActorX" 
    },
    "R": "ActorX appeared in a film that won Best Picture."
}

Now you need to directly output the results of the following question in JSON format (must include "A" and "R") without other information or notes. If the triplets explicitly contains the answer to the question, prioritize the fact of the triplet over memory.
Q: """

judge_reverse = """Based on the current set of entities to be retrieved and the known information including memory and triplets, is it necessary to add additional entities for answering question?
Here are two examples:
Q: Which of the countries in the Caribbean has the smallest country calling code?
Entities set to be retrieved: ['Anguilla', 'Saint Lucia']
Memory: Caribbean contains Antilles and Saint Lucia
Knowledge Triplets: Caribbean, location.location.contains, ['Antilles', 'Saint Lucia']
Output: 
{
    "Add": "Yes",
    "Reason": "The entities set ignores other countries in Caribbean."
}

Q: The artist nominated for The Long Winter lived where?
Entities set to be retrieved: ['Laura Ingalls Wilder']
Memory: "The author of The Long Winter is Laura Ingalls Wilder."
Knowledge Triplets: The Long Winter, book.written_work.author, [Laura Ingalls Wilder]
Output: 
{
    "Add": "No",
    "Reason": "Now you need to search where Laura Ingalls Wilder lived."
}

Now you need to directly output the results of the following question in the JSON format (must include "Add" and "Reason") without other information or notes.
Q: """


add_ent_prompt = """Please select the fewest necessary entities to be retrieved for answering the Q from Candidate Entities, based on the current known information (Memory), the reason for additional retrieval, and your own knowledge.
Here is an example:
Q: Which of the countries in the Caribbean has the smallest country calling code?
Reason: The entities set ignores other countries in the Caribbean.
Candidate Entities: ['Saint Marie', 'Saint Martin (Island)', 'Viceroy Anguilla', 'Lesser Antilles', 'Barbados', 'British Virgin Islands', 'Leeward Islands', 'British West Indies', 'Caribbean', 'Saint Thomas', 'Bronte International University', 'Collectivity of Saint Martin', 'Southern Caribbean', 'University of Medicine and Health Sciences', 'Soufrière Hills', 'Lucayan Archipelago', 'University of the West Indies', 'Aureus University School of Medicine', 'North America', 'Netherlands Antilles', 'Puerto Rico', 'Chances Peak', 'Clarendon Parish', 'Saint Kitts and Nevis', 'Saint Lucia', 'Americas', 'Caribbean special municipalities of the Netherlands', 'Sandy Hill', 'School of Business and Computer Science, Trincity', 'School of Business and Computer Science, San Fernando', 'School of Business and Computer Science, Champs Fleurs', 'School of Business and Computer Science, Port of Spain', 'Bridgetown', 'St. Martinus University School of Medicine, main campus', 'Higher Institute of Medical Sciences. main campus', 'Grace University, main campus', 'Anguilla']
Memory: {
    "1": "The countries in the Caribbean are Antilles and Latin America.",
    "2": "The country calling codes for Antilles and Latin America are not mentioned.",
    "3": "It is not mentioned which country has the smallest country calling code."
}
Output: ['Barbados', 'Saint Lucia', 'Anguilla']

Now you need to directly output the results for the following Q in the list format without other information or notes.
Q: """


cot_prompt = """Please answer the question according to your knowledge step by step. Here is an example:
Q: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
The output is:
{
    "A": {
        "Known": "Yes",
        "Answer": "Washington, D.C."
    },
    "R": "First, the education institution has a sports team named George Washington Colonials men's basketball in is George Washington University , Second, George Washington University is in Washington D.C."
}

Please directly output the answer in JSON format (must include "A" and "R") without other information or notes.
"""

speculate_prompt = """Task: Predict the logical relation path (Skeleton) for the new question.

[Instructions]
1. Use the provided examples ONLY as a format reference.
2. CRITICAL: Your prediction must be based on the SEMANTICS of the [New Question], NOT the topic of the examples.
3. If the examples are irrelevant to the new question, IGNORE them and use your own knowledge.

[Example 1]
Q: {q1}
Output: {s1}

[Example 2]
Q: {q2}
Output: {s2}

[New Question]
Q: {question}
Output: """




