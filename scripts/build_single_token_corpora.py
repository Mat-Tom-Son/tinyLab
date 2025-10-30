#!/usr/bin/env python3
"""Generate four single-token corpora for cross-condition experiments."""

import itertools
import json
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

from transformers import AutoTokenizer


TOKENIZER = AutoTokenizer.from_pretrained("gpt2")
CORPUS_ROOT = Path("lab/data/corpora")
EXAMPLES_PER_CORPUS = 200


def is_single_token(text: str) -> bool:
    return len(TOKENIZER.encode(text, add_special_tokens=False)) == 1


def ensure_single(texts: Sequence[str], context: str) -> None:
    bad = [t for t in texts if not is_single_token(t)]
    if bad:
        raise ValueError(f"{context}: non single-token entries detected: {bad[:5]}")


def pairwise_examples(items: Sequence[Tuple[str, str]]) -> List[Tuple[str, str, str, str]]:
    """Build (clean, corrupt, target, foil) by pairing prompts from the item set."""
    examples: List[Tuple[str, str, str, str]] = []
    for i, (prompt_a, target_a) in enumerate(items):
        for j, (prompt_b, target_b) in enumerate(items):
            if i == j:
                continue
            examples.append((prompt_a, prompt_b, target_a, target_b))
    return examples


def round_robin(lists: Sequence[List[Tuple[str, str, str, str]]]) -> Iterator[Tuple[str, str, str, str]]:
    queues = [list(lst) for lst in lists if lst]
    idx = 0
    while queues:
        if idx >= len(queues):
            idx = 0
        if not queues[idx]:
            queues.pop(idx)
            continue
        yield queues[idx].pop(0)
        idx += 1


def build_fact_candidates() -> List[Tuple[str, str, str, str]]:
    colors = [
        ("The color of grass is", " green"),
        ("The color of coal is", " black"),
        ("The color of snow is", " white"),
        ("The color of lemons is", " yellow"),
        ("The color of pumpkins is", " orange"),
        ("The color of lavender is", " purple"),
        ("The color of sand is", " brown"),
        ("The color of the sky is", " blue"),
        ("The color of clouds is", " gray"),
        ("The color of steel is", " silver"),
        ("The color of gold is", " gold"),
        ("The color of clay is", " red"),
    ]
    ensure_single([color for _, color in colors], "facts: colors")

    animals = [
        ("The animal that barks is the", " dog"),
        ("The animal that purrs is the", " cat"),
        ("The animal that howls is the", " wolf"),
        ("The animal that roars is the", " lion"),
        ("The animal that hoots is the", " owl"),
        ("The animal that gallops is the", " horse"),
        ("The animal that gives wool is the", " sheep"),
        ("The animal that climbs cliffs is the", " goat"),
        ("The animal that quacks is the", " duck"),
        ("The animal that produces milk is the", " cow"),
        ("The insect that makes honey is the", " bee"),
        ("The insect that builds colonies is the", " ant"),
    ]
    ensure_single([target for _, target in animals], "facts: animals")

    body = [
        ("The organ that pumps blood is the", " heart"),
        ("The organ that controls thought is the", " brain"),
        ("The organ that absorbs oxygen is the", " lung"),
        ("The organ that filters toxins is the", " liver"),
        ("The organ that filters blood is the", " kidney"),
        ("The organ that digests food is the", " stomach"),
        ("The organ that protects the body is the", " skin"),
        ("The fluid that carries oxygen is", " blood"),
        ("The organ that sees light is the", " eye"),
        ("The organ that hears sound is the", " ear"),
        ("The organ that smells is the", " nose"),
        ("The organ that tastes is the", " mouth"),
    ]
    ensure_single([target for _, target in body], "facts: body")

    food = [
        ("The fruit that is red is the", " apple"),
        ("The fruit that is yellow is the", " banana"),
        ("The fruit that is orange is the", " mango"),
        ("The fruit that is green is the", " pear"),
        ("The fruit that is purple is the", " plum"),
        ("The fruit that grows in clusters is the", " grape"),
        ("The fruit that is sour and yellow is the", " lemon"),
        ("The fruit that is tart and green is the", " lime"),
        ("The fruit that has fuzzy skin is the", " peach"),
        ("The drink made from roasted beans is", " coffee"),
        ("The drink made from leaves is", " tea"),
        ("The grain used for bread is", " wheat"),
    ]
    ensure_single([target for _, target in food], "facts: food")

    math = [
        ("Two plus two equals", " 4"),
        ("Three plus three equals", " 6"),
        ("Four times two equals", " 8"),
        ("Five plus five equals", " 10"),
        ("Nine minus three equals", " 6"),
        ("Six minus four equals", " 2"),
        ("Eight divided by two equals", " 4"),
        ("Ten minus seven equals", " 3"),
        ("Seven plus one equals", " 8"),
        ("Nine minus five equals", " 4"),
        ("Six plus three equals", " 9"),
        ("Five minus two equals", " 3"),
    ]
    ensure_single([ans for _, ans in math], "facts: math")

    geography = [
        ("The capital of France is", " Paris"),
        ("The capital of Italy is", " Rome"),
        ("The capital of Germany is", " Berlin"),
        ("The capital of Spain is", " Madrid"),
        ("The capital of Austria is", " Vienna"),
        ("The capital of Sweden is", " Stockholm"),
        ("The capital of Norway is", " Oslo"),
        ("The capital of Finland is", " Helsinki"),
        ("The capital of Russia is", " Moscow"),
        ("The capital of Japan is", " Tokyo"),
        ("The capital of China is", " Beijing"),
        ("The capital of Canada is", " Ottawa"),
    ]
    ensure_single([target for _, target in geography], "facts: geography")

    weather = [
        ("The season with snow is", " winter"),
        ("The season with rain showers is", " spring"),
        ("The season with heatwaves is", " summer"),
        ("The season with falling leaves is", " autumn"),
        ("Frozen water from the sky is called", " snow"),
        ("Water falling from the sky is called", " rain"),
        ("Moving air is called", " wind"),
        ("A violent rotating storm is a", " tornado"),
        ("A tropical storm over oceans is a", " cyclone"),
        ("The measure of heat is called", " temperature"),
    ]
    weather = [item for item in weather if is_single_token(item[1])]

    categories = [
        pairwise_examples(colors),
        pairwise_examples(animals),
        pairwise_examples(body),
        pairwise_examples(food),
        pairwise_examples(math),
        pairwise_examples(geography),
        pairwise_examples(weather),
    ]

    ordered = list(itertools.islice(round_robin(categories), EXAMPLES_PER_CORPUS * 2))
    return ordered[:400]


def make_negation_pairs() -> List[Tuple[str, str]]:
    sentences = set()

    capitals = [
        ("Paris", "the capital of France"),
        ("Rome", "the capital of Italy"),
        ("Berlin", "the capital of Germany"),
        ("Madrid", "the capital of Spain"),
        ("Vienna", "the capital of Austria"),
        ("Stockholm", "the capital of Sweden"),
        ("Oslo", "the capital of Norway"),
        ("Helsinki", "the capital of Finland"),
        ("Moscow", "the capital of Russia"),
        ("Tokyo", "the capital of Japan"),
        ("Beijing", "the capital of China"),
        ("Ottawa", "the capital of Canada"),
        ("Cairo", "the capital of Egypt"),
        ("Athens", "the capital of Greece"),
        ("Warsaw", "the capital of Poland"),
        ("Prague", "the capital of the Czech Republic"),
        ("Lisbon", "the capital of Portugal"),
        ("Dublin", "the capital of Ireland"),
        ("Bern", "the capital of Switzerland"),
        ("Seoul", "the capital of South Korea"),
        ("Manila", "the capital of the Philippines"),
        ("Jakarta", "the capital of Indonesia"),
        ("Riyadh", "the capital of Saudi Arabia"),
        ("Tehran", "the capital of Iran"),
        ("Baghdad", "the capital of Iraq"),
        ("Ankara", "the capital of Turkey"),
        ("Pretoria", "the capital of South Africa"),
        ("Accra", "the capital of Ghana"),
        ("Lima", "the capital of Peru"),
        ("Santiago", "the capital of Chile"),
        ("Quito", "the capital of Ecuador"),
        ("La Paz", "the capital of Bolivia"),
        ("Montevideo", "the capital of Uruguay"),
        ("Asuncion", "the capital of Paraguay"),
        ("Sucre", "the capital of Bolivia"),
        ("Bogota", "the capital of Colombia"),
        ("Caracas", "the capital of Venezuela"),
        ("Brasilia", "the capital of Brazil")
    ]

    for subject, predicate in capitals:
        sentences.add(f"{subject} is {predicate}")
        sentences.add(f"It is true that {subject} is {predicate}")

    categories = {
        "animals": [
            ("Dogs", "mammals"),
            ("Cats", "mammals"),
            ("Birds", "animals"),
            ("Fish", "animals"),
            ("Bees", "insects"),
            ("Cows", "mammals"),
            ("Goats", "herbivores"),
            ("Sheep", "grazers"),
            ("Owls", "hunters"),
            ("Hawks", "birds"),
            ("Penguins", "birds"),
            ("Sharks", "predators"),
            ("Whales", "mammals"),
            ("Spiders", "arachnids"),
            ("Ants", "workers"),
            ("Bats", "mammals"),
            ("Seals", "divers"),
            ("Otters", "hunters"),
            ("Lions", "predators"),
            ("Tigers", "predators")
        ],
        "plants": [
            ("Flowers", "plants"),
            ("Trees", "plants"),
            ("Leaves", "green"),
            ("Roots", "anchors"),
            ("Seeds", "sprouts"),
            ("Grass", "greenery"),
            ("Vines", "climbers"),
            ("Moss", "plants"),
            ("Ferns", "plants"),
            ("Cacti", "succulents")
        ],
        "food": [
            ("Apples", "fruit"),
            ("Bananas", "fruit"),
            ("Grapes", "fruit"),
            ("Peaches", "fruit"),
            ("Plums", "fruit"),
            ("Lemons", "sour"),
            ("Limes", "tart"),
            ("Carrots", "orange"),
            ("Tomatoes", "red"),
            ("Potatoes", "tubers"),
            ("Onions", "vegetables"),
            ("Garlic", "bulbs"),
            ("Peppers", "vegetables"),
            ("Spinach", "greens"),
            ("Lettuce", "greens")
        ],
        "elements": [
            ("Water", "wet"),
            ("Fire", "hot"),
            ("Ice", "cold"),
            ("Smoke", "gray"),
            ("Salt", "salty"),
            ("Sugar", "sweet"),
            ("Coffee", "bitter"),
            ("Tea", "warm"),
            ("Bread", "baked"),
            ("Rain", "wet"),
            ("Snow", "cold"),
            ("Wind", "moving"),
            ("Clouds", "vapor"),
            ("Thunder", "loud"),
            ("Lightning", "bright"),
            ("Storms", "windy"),
            ("Sunlight", "bright"),
            ("Night", "dark"),
            ("Day", "bright"),
            ("Stone", "hard"),
            ("Wood", "flammable"),
            ("Metal", "strong"),
            ("Glass", "transparent"),
            ("Clay", "malleable")
        ],
        "people": [
            ("Singers", "artists"),
            ("Dancers", "performers"),
            ("Actors", "artists"),
            ("Teachers", "guides"),
            ("Doctors", "experts"),
            ("Nurses", "helpers"),
            ("Poets", "writers"),
            ("Chefs", "cooks"),
            ("Farmers", "growers"),
            ("Miners", "workers"),
            ("Builders", "makers"),
            ("Drivers", "handlers"),
            ("Pilots", "flyers"),
            ("Sailors", "travelers"),
            ("Gamers", "players"),
            ("Readers", "thinkers"),
            ("Leaders", "guides"),
            ("Artists", "creatives"),
            ("Scientists", "researchers"),
            ("Engineers", "designers")
        ]
    }

    for group in categories.values():
        for subject, predicate in group:
            sentences.add(f"{subject} are {predicate}")
            sentences.add(f"All {subject.lower()} are {predicate}")

    # Derive additional statements from the factual corpus templates
    fact_examples = build_fact_candidates()
    for clean, _, target, _ in fact_examples:
        statement = f"{clean} {target.strip()}"
        sentences.add(statement)

    def make_pair(sentence: str) -> Tuple[str, str]:
        if " is " in sentence:
            return sentence, sentence.replace(" is ", " is not ", 1)
        if " are " in sentence:
            return sentence, sentence.replace(" are ", " are not ", 1)
        return sentence, f"{sentence} is not true"

    pairs = []
    for sentence in sentences:
        clean, corrupt = make_pair(sentence)
        if clean != corrupt:
            pairs.append((clean, corrupt))

    return pairs


def build_counterfactual_candidates() -> List[Tuple[str, str, str, str]]:
    def make(clean_ant: str, corrupt_ant: str, suffix: str, target: str, foil: str) -> Tuple[str, str, str, str]:
        clean_stmt = f"If {clean_ant}, then{suffix} {target.strip()}"
        corrupt_stmt = f"If {corrupt_ant}, then{suffix} {foil.strip()}"
        return clean_stmt, corrupt_stmt, target, foil

    animal_rules = [
        make("cats are mammals", "rocks are mammals", " they have", " fur", " scales"),
        make("dogs are loyal", "stones are loyal", " they show", " care", " rust"),
        make("birds can fly", "rocks can fly", " they use", " wings", " roots"),
        make("fish can swim", "trees can swim", " they use", " fins", " roots"),
        make("bees make honey", "clouds make honey", " they produce", " honey", " vapor"),
        make("cows chew grass", "cars chew grass", " they chew", " grass", " metal"),
        make("wolves hunt prey", "flowers hunt prey", " they chase", " prey", " pollen"),
        make("lions roar loudly", "leaves roar loudly", " they sound", " loud", " quiet"),
        make("owls see at night", "pebbles see at night", " they use", " eyes", " dust"),
        make("bats navigate caves", "chairs navigate caves", " they use", " echo", " wood"),
        make("horses gallop fast", "pillars gallop fast", " they move", " fast", " still"),
        make("sheep grow wool", "clouds grow wool", " they grow", " wool", " vapor"),
        make("goats climb cliffs", "waters climb cliffs", " they use", " legs", " waves"),
        make("ducks swim well", "stones swim well", " they float on", " water", " dust"),
        make("geese migrate yearly", "tables migrate yearly", " they travel in", " group", " silence"),
        make("frogs leap high", "rocks leap high", " they jump", " high", " still"),
        make("snakes slither", "bricks slither", " they move", " smooth", " rigid"),
        make("turtles wear shells", "trees wear shells", " they carry", " shells", " bark"),
        make("whales breathe air", "pebbles breathe air", " they need", " air", " stone"),
        make("sharks hunt fish", "flowers hunt fish", " they hunt", " fish", " pollen"),
    ]

    nature_rules = [
        make("plants need light", "rocks need light", " they need", " light", " shade"),
        make("trees grow leaves", "statues grow leaves", " they grow", " leaves", " dust"),
        make("seeds sprout", "coins sprout", " they become", " plants", " metal"),
        make("glaciers melt", "fires melt", " they become", " water", " ash"),
        make("snow melts", "embers melt", " it becomes", " water", " dust"),
        make("rain falls", "smoke falls", " it makes", " water", " dust"),
        make("sun rises daily", "rocks rise daily", " the sky turns", " bright", " dim"),
        make("moon shines nightly", "sand shines nightly", " the night is", " light", " dark"),
        make("wind blows", "walls blow", " the air feels", " cool", " hard"),
        make("waves crash", "hills crash", " the shore becomes", " wet", " dry"),
        make("fire burns fuel", "ice burns fuel", " it gives", " heat", " frost"),
        make("ice freezes water", "flames freeze water", " it turns to", " ice", " smoke"),
        make("metal rusts", "clouds rust", " it turns", " red", " blue"),
        make("clay hardens", "steam hardens", " it becomes", " solid", " vapor"),
        make("fog lifts", "stones lift", " the view is", " clear", " hidden"),
        make("storms gather", "books gather", " the clouds grow", " dark", " dull"),
        make("volcanoes erupt", "rivers erupt", " they release", " lava", " water"),
        make("magnets attract metal", "feathers attract metal", " they pull", " metal", " dust"),
        make("lightning flashes", "puddles flash", " the sky grows", " bright", " dim"),
        make("thunder rumbles", "petals rumble", " the air becomes", " loud", " quiet"),
    ]

    tool_rules = [
        make("alarms ring", "pillows ring", " people", " wake", " sleep"),
        make("clocks tick", "clouds tick", " they measure", " time", " vapor"),
        make("engines run", "caves run", " they need", " fuel", " stone"),
        make("circuits close", "mirrors close", " the current will", " flow", " stop"),
        make("switches flip", "rivers flip", " the lights can", " turn", " flood"),
        make("rockets launch", "roots launch", " they reach", " space", " soil"),
        make("mirrors reflect", "smoke reflects", " they show", " images", " fumes"),
        make("cameras record", "winds record", " they capture", " light", " noise"),
        make("keys unlock", "leaves unlock", " they open the", " door", " dust"),
        make("locks jam", "streams jam", " the gate stays", " shut", " open"),
        make("bridges span rivers", "clouds span rivers", " they cross the", " river", " sky"),
        make("tunnels dig earth", "storms dig earth", " they pass through", " soil", " sky"),
        make("gears turn", "shadows turn", " machines will", " move", " still"),
        make("levers lift loads", "petals lift loads", " they raise the", " load", " scent"),
        make("pulleys hoist loads", "mirrors hoist loads", " they raise a", " load", " glare"),
        make("ropes pull loads", "beams pull loads", " they pull the", " load", " dust"),
        make("boats float", "stones float", " they stay on", " water", " dust"),
        make("planes fly", "logs fly", " they travel through", " air", " mud"),
        make("cars roll", "trees roll", " they move on", " wheels", " roots"),
        make("trains glide", "clouds glide", " they ride on", " tracks", " wind"),
    ]

    people_rules = [
        make("bikes balance", "rocks balance", " they use", " wheels", " dust"),
        make("skaters glide", "statues glide", " they move", " smooth", " still"),
        make("singers sing", "stones sing", " they use their", " voice", " dust"),
        make("writers write", "clouds write", " they use a", " pen", " mist"),
        make("artists paint", "waves paint", " they use", " color", " foam"),
        make("chefs cook", "pages cook", " they use", " heat", " paper"),
        make("farmers plant", "storms plant", " they plant", " seeds", " dust"),
        make("miners dig", "breezes dig", " they use a", " tool", " gust"),
        make("builders build", "rivers build", " they lift", " stone", " water"),
        make("teachers teach", "rocks teach", " they share", " facts", " silence"),
        make("doctors heal", "shadows heal", " they give", " care", " cold"),
        make("nurses help", "sparks help", " they offer", " aid", " smoke"),
        make("pilots fly", "roots fly", " they steer", " plane", " soil"),
        make("sailors sail", "mountains sail", " they guide", " ship", " cliffs"),
        make("captains steer", "clouds steer", " they hold the", " wheel", " mist"),
        make("drivers steer", "stones steer", " they turn the", " wheel", " dust"),
        make("walkers walk", "pillars walk", " they use their", " feet", " stone"),
        make("runners sprint", "chairs sprint", " they move", " fast", " still"),
        make("climbers climb", "rivers climb", " they use their", " hands", " waves"),
        make("divers dive", "fires dive", " they enter", " water", " flame"),
        make("swimmers swim", "clouds swim", " they move through", " water", " air"),
        make("dancers dance", "rocks dance", " they move with", " grace", " grit"),
        make("actors act", "storms act", " they play a", " role", " gust"),
        make("leaders lead", "waves lead", " they guide the", " team", " spray"),
        make("readers read", "stones read", " they turn the", " page", " dust"),
        make("speakers speak", "mountains speak", " they share", " words", " silence"),
        make("thinkers think", "clouds think", " they form", " ideas", " vapor"),
        make("inventors invent", "rivers invent", " they build", " tools", " current"),
        make("scientists test", "storms test", " they run", " trials", " thunder"),
        make("artists sketch", "shadows sketch", " they draw", " lines", " smoke"),
        make("gardeners water", "stones water", " they pour", " water", " sand"),
        make("bakers bake", "winds bake", " they heat", " bread", " air"),
        make("tailors sew", "waves sew", " they stitch", " cloth", " foam"),
        make("poets write", "rivers write", " they craft", " poems", " river"),
        make("students learn", "rocks learn", " they gain", " ideas", " silence"),
        make("musicians play", "mountains play", " they play", " music", " stone"),
        make("painters paint", "clouds paint", " they paint", " lines", " mist"),
        make("drivers stop", "clouds stop", " they press the", " brake", " breeze"),
        make("pilots land", "rivers land", " they land the", " plane", " stream"),
        make("sailors dock", "storms dock", " they tie the", " ship", " gust"),
        make("miners blast", "waves blast", " they break the", " rock", " foam"),
        make("farmers harvest", "thunder harvests", " they collect the", " crops", " roar"),
        make("chefs season", "shadows season", " they add", " spice", " dusk"),
        make("writers edit", "winds edit", " they change", " words", " gust"),
        make("coders code", "rivers code", " they write", " logic", " current"),
        make("gamers play", "stones play", " they enjoy", " games", " dust"),
        make("singers harmonize", "storms harmonize", " they blend", " voice", " thunder"),
        make("readers imagine", "pillars imagine", " they picture", " scenes", " stone"),
        make("gardeners prune", "clouds prune", " they trim", " plants", " mist"),
        make("teachers guide", "mountains guide", " they lead the", " class", " cliff"),
        make("doctors diagnose", "shadows diagnose", " they find", " causes", " dusk"),
        make("nurses monitor", "waves monitor", " they check", " vital", " foam"),
        make("pilots navigate", "rocks navigate", " they follow", " routes", " dust"),
        make("captains command", "grains command", " they lead the", " crew", " sand"),
        make("leaders inspire", "storms inspire", " they spark", " hope", " wind"),
        make("thinkers reason", "mountains reason", " they shape", " logic", " stone"),
    ]

    all_rules = animal_rules + nature_rules + tool_rules + people_rules

    # Derive additional conditional rules from the factual dataset
    fact_examples = build_fact_candidates()
    for clean, corrupt, target, foil in fact_examples:
        antecedent_true = f"{clean} {target.strip()}"
        antecedent_false = f"{corrupt} {target.strip()}"
        clean_stmt = f"If {antecedent_true}, then the answer is {target.strip()}"
        corrupt_stmt = f"If {antecedent_false}, then the answer is {foil.strip()}"
        all_rules.append((clean_stmt, corrupt_stmt, target, foil))

    # Deduplicate while preserving order
    seen = set()
    deduped_rules = []
    for rule in all_rules:
        key = tuple(rule)
        if key not in seen:
            seen.add(key)
            deduped_rules.append(rule)

    ensure_single([rule[2] for rule in deduped_rules] + [rule[3] for rule in deduped_rules], "counterfactual targets/foils")
    return deduped_rules[:300]
def build_logical_candidates() -> List[Tuple[str, str, str, str]]:
    true_statements = [
        "Dogs are mammals",
        "Cats are animals",
        "Birds have wings",
        "Fish can swim",
        "Bees make honey",
        "Cows produce milk",
        "Owls hunt at night",
        "Wolves live in packs",
        "Lions roar loudly",
        "Foxes hunt quietly",
        "Bears hibernate",
        "Frogs can jump",
        "Snakes slither",
        "Turtles have shells",
        "Ducks can swim",
        "Geese migrate",
        "Sheep graze",
        "Goats climb",
        "Hawks soar",
        "Eagles fly",
        "Grass is green",
        "Snow is cold",
        "Fire is hot",
        "Rain is wet",
        "Ice can melt",
        "Clouds hold vapor",
        "Wind can blow",
        "Sunlight is bright",
        "Night is dark",
        "Day is light",
        "Water is liquid",
        "Stone is hard",
        "Wood can burn",
        "Metal can rust",
        "Plants need light",
        "Seeds can sprout",
        "Flowers can bloom",
        "Leaves grow",
        "Trees absorb water",
        "Roots hold soil",
        "Apples grow on trees",
        "Bananas grow in bunches",
        "Grapes grow on vines",
        "Peaches have pits",
        "Plums have pits",
        "Lemons are sour",
        "Limes are tart",
        "Carrots are orange",
        "Tomatoes are red",
        "Cats have whiskers",
        "Dogs have tails",
        "Birds lay eggs",
        "Fish have fins",
        "Trees have roots",
        "Humans can think",
        "Rain falls",
        "Fire burns fuel",
        "Snow melts",
        "Ice is cold",
    ]

    false_statements = [
        "Dogs lay eggs",
        "Cats bark loudly",
        "Birds crawl slowly",
        "Fish climb trees",
        "Bees make bricks",
        "Cows breathe water",
        "Owls sleep at noon",
        "Wolves eat rocks",
        "Lions whisper",
        "Foxes live underwater",
        "Bears fly south",
        "Frogs drive cars",
        "Snakes walk upright",
        "Turtles sprint quickly",
        "Ducks speak words",
        "Geese dig tunnels",
        "Sheep hunt prey",
        "Goats swim oceans",
        "Hawks sleep underwater",
        "Eagles crawl slowly",
        "Grass glows purple",
        "Snow burns skin",
        "Fire freezes water",
        "Rain stays dry",
        "Ice boils",
        "Clouds fall like stones",
        "Wind sits still",
        "Sunlight is dark",
        "Night is bright",
        "Day is silent",
        "Water grows hair",
        "Stone melts easily",
        "Wood turns to steel",
        "Metal tastes sweet",
        "Plants hate light",
        "Seeds stay asleep",
        "Flowers sink",
        "Leaves sing songs",
        "Trees fly away",
        "Roots float away",
        "Apples grow underground",
        "Bananas glow blue",
        "Grapes bark",
        "Peaches walk",
        "Plums shout",
        "Lemons sleep",
        "Limes roar",
        "Carrots fly",
        "Tomatoes freeze fire",
        "Cats drive cars",
        "Dogs speak words",
        "Birds breathe water",
        "Fish walk deserts",
        "Trees sing songs",
        "Humans never think",
        "Rain stays dry forever",
        "Fire freezes metal",
        "Snow boils water",
        "Ice burns skin",
    ]

    operators = ["AND", "OR", "XOR", "IMPLIES"]

    def render(op: str, a: str, b: str) -> str:
        if op == "IMPLIES":
            return f"{a} IMPLIES {b}"
        return f"{a} {op} {b}"

    def eval_op(op: str, val_a: bool, val_b: bool) -> bool:
        if op == "AND":
            return val_a and val_b
        if op == "OR":
            return val_a or val_b
        if op == "XOR":
            return val_a != val_b
        if op == "IMPLIES":
            return (not val_a) or val_b
        raise ValueError(op)

    scenarios = []
    for i, a in enumerate(true_statements):
        b = false_statements[i]
        scenarios.append((a, True, b, False))
    for i, a in enumerate(true_statements):
        b = true_statements[(i + 7) % len(true_statements)]
        scenarios.append((a, True, b, True))
    for i, a in enumerate(false_statements):
        b = false_statements[(i + 5) % len(false_statements)]
        scenarios.append((a, False, b, False))
    for i, a in enumerate(false_statements):
        b = true_statements[(i + 9) % len(true_statements)]
        scenarios.append((a, False, b, True))

    candidates: List[Tuple[str, str, str, str]] = []
    for a_sent, a_val, b_sent, b_val in scenarios:
        truth_map = {op: eval_op(op, a_val, b_val) for op in operators}
        true_ops = [op for op, value in truth_map.items() if value]
        false_ops = [op for op, value in truth_map.items() if not value]
        if not true_ops or not false_ops:
            continue
        clean_op = true_ops[0]
        corrupt_op = false_ops[0]
        clean = render(clean_op, a_sent, b_sent)
        corrupt = render(corrupt_op, a_sent, b_sent)
        target = " true" if truth_map[clean_op] else " false"
        foil = " false" if target == " true" else " true"
        candidates.append((clean, corrupt, target, foil))
        if len(candidates) >= 200:
            break
    return candidates


def write_jsonl(path: Path, rows: List[Tuple[str, str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for clean, corrupt, target, foil in rows:
            f.write(
                json.dumps(
                    {
                        "clean": clean,
                        "corrupt": corrupt,
                        "target": target,
                        "foil": foil,
                    }
                )
            )
            f.write("\n")


def main() -> None:
    facts = build_fact_candidates()[:EXAMPLES_PER_CORPUS]
    neg_pairs = make_negation_pairs()[:EXAMPLES_PER_CORPUS]
    negation = [(clean, corrupt, " true", " false") for clean, corrupt in neg_pairs]
    counterfactual = build_counterfactual_candidates()[:EXAMPLES_PER_CORPUS]
    logical = build_logical_candidates()[:EXAMPLES_PER_CORPUS]

    write_jsonl(CORPUS_ROOT / "facts_single_token_v1.jsonl", facts)
    write_jsonl(CORPUS_ROOT / "negation_single_token_v1.jsonl", negation)
    write_jsonl(CORPUS_ROOT / "counterfactual_single_token_v1.jsonl", counterfactual)
    write_jsonl(CORPUS_ROOT / "logical_single_token_v1.jsonl", logical)
    print("Generated corpora: facts, negation, counterfactual, logical")


if __name__ == "__main__":
    main()
