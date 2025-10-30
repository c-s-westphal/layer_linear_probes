#!/usr/bin/env python3
"""
Linear Probe PCA Experiment

Trains linear probes on PCA-reduced activations from each layer of GPT-2 Small
to predict linguistic features. Measures mutual information and classification
accuracy across layers.

Tasks:
1. Plurality (Binary): Predict if noun is singular or plural
2. Part of Speech (4-class): Predict POS tag (noun/verb/adjective/adverb)

For each layer (1-11, skipping layer 0 input embeddings):
- Extract 768-dim activations at target token positions
- Apply PCA to reduce to 10 principal components
- Train 3 LogisticRegression probes (for confidence intervals)
- Measure mutual information, accuracy, and F1 score
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mutual_info_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.model import ModelLoader
from pos_dataset_generator import generate_pos_dataset


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to both file and console."""
    logger = logging.getLogger("linear_probe_pca")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fh = logging.FileHandler(output_dir / "experiment.log")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def create_plurality_dataset() -> List[Dict]:
    """
    Create dataset for plurality prediction task with 500 unique examples each.

    Returns:
        List of 1000 examples with 'text', 'target_word', and 'label' (0=singular, 1=plural)
    """
    # Generate 500 unique singular examples
    singular_examples = [
        # Animals (50 examples)
        ("The cat sits on the windowsill.", "cat"),
        ("A dog barks at strangers.", "dog"),
        ("The bird sings in the morning.", "bird"),
        ("A horse gallops across the field.", "horse"),
        ("The rabbit hops through the garden.", "rabbit"),
        ("A lion roars in the jungle.", "lion"),
        ("The elephant walks slowly.", "elephant"),
        ("A tiger hunts at night.", "tiger"),
        ("The monkey swings from trees.", "monkey"),
        ("A dolphin swims gracefully.", "dolphin"),
        ("The penguin waddles on ice.", "penguin"),
        ("A bear hibernates in winter.", "bear"),
        ("The snake slithers quietly.", "snake"),
        ("A frog jumps into the pond.", "frog"),
        ("The butterfly emerges from its cocoon.", "butterfly"),
        ("A bee buzzes around flowers.", "bee"),
        ("The ant carries heavy loads.", "ant"),
        ("A spider spins its web.", "spider"),
        ("The fish swims upstream.", "fish"),
        ("A shark patrols the waters.", "shark"),
        ("The whale breaches the surface.", "whale"),
        ("An eagle soars above mountains.", "eagle"),
        ("The owl hoots at midnight.", "owl"),
        ("A parrot repeats words.", "parrot"),
        ("The crow caws loudly.", "crow"),
        ("A seagull flies over the ocean.", "seagull"),
        ("The duck quacks in the pond.", "duck"),
        ("A goose honks aggressively.", "goose"),
        ("The turkey gobbles nervously.", "turkey"),
        ("A chicken pecks at grain.", "chicken"),
        ("The cow moos in the barn.", "cow"),
        ("A pig wallows in mud.", "pig"),
        ("The sheep grazes on grass.", "sheep"),
        ("A goat climbs the rocks.", "goat"),
        ("The donkey brays stubbornly.", "donkey"),
        ("A camel travels through deserts.", "camel"),
        ("The giraffe reaches high branches.", "giraffe"),
        ("A zebra has distinctive stripes.", "zebra"),
        ("The rhino charges forward.", "rhino"),
        ("A hippo rests in water.", "hippo"),
        ("The kangaroo jumps far.", "kangaroo"),
        ("A koala sleeps in trees.", "koala"),
        ("The panda eats bamboo.", "panda"),
        ("A wolf howls at the moon.", "wolf"),
        ("The fox hunts cleverly.", "fox"),
        ("A deer runs through the forest.", "deer"),
        ("The moose has large antlers.", "moose"),
        ("A raccoon searches for food.", "raccoon"),
        ("The squirrel collects nuts.", "squirrel"),
        ("A mouse scurries away.", "mouse"),

        # People and Professions (100 examples)
        ("The student studies diligently.", "student"),
        ("A teacher explains concepts clearly.", "teacher"),
        ("The professor lectures enthusiastically.", "professor"),
        ("A doctor examines patients carefully.", "doctor"),
        ("The nurse administers medication.", "nurse"),
        ("A surgeon performs operations.", "surgeon"),
        ("The dentist cleans teeth.", "dentist"),
        ("A therapist listens attentively.", "therapist"),
        ("The scientist conducts experiments.", "scientist"),
        ("A researcher analyzes data.", "researcher"),
        ("The engineer designs systems.", "engineer"),
        ("A programmer writes code.", "programmer"),
        ("The developer builds applications.", "developer"),
        ("A designer creates graphics.", "designer"),
        ("The artist paints masterpieces.", "artist"),
        ("A musician plays instruments.", "musician"),
        ("The singer performs songs.", "singer"),
        ("A dancer moves gracefully.", "dancer"),
        ("The actor portrays characters.", "actor"),
        ("A director manages productions.", "director"),
        ("The writer composes stories.", "writer"),
        ("An author publishes books.", "author"),
        ("The poet crafts verses.", "poet"),
        ("A journalist reports news.", "journalist"),
        ("The editor reviews manuscripts.", "editor"),
        ("A photographer captures images.", "photographer"),
        ("The chef prepares meals.", "chef"),
        ("A cook follows recipes.", "cook"),
        ("The waiter serves customers.", "waiter"),
        ("A bartender mixes drinks.", "bartender"),
        ("The lawyer argues cases.", "lawyer"),
        ("An attorney represents clients.", "attorney"),
        ("The judge presides over trials.", "judge"),
        ("A politician campaigns actively.", "politician"),
        ("The mayor governs the city.", "mayor"),
        ("A senator proposes legislation.", "senator"),
        ("The officer patrols the streets.", "officer"),
        ("A detective solves crimes.", "detective"),
        ("The firefighter rescues people.", "firefighter"),
        ("A paramedic provides emergency care.", "paramedic"),
        ("The soldier follows orders.", "soldier"),
        ("A pilot flies aircraft.", "pilot"),
        ("The captain commands the ship.", "captain"),
        ("An astronaut explores space.", "astronaut"),
        ("The mechanic repairs vehicles.", "mechanic"),
        ("A plumber fixes pipes.", "plumber"),
        ("The electrician installs wiring.", "electrician"),
        ("A carpenter builds furniture.", "carpenter"),
        ("The architect plans buildings.", "architect"),
        ("A contractor manages construction.", "contractor"),
        ("The farmer grows crops.", "farmer"),
        ("A gardener tends plants.", "gardener"),
        ("The fisherman catches fish.", "fisherman"),
        ("A miner extracts minerals.", "miner"),
        ("The librarian organizes books.", "librarian"),
        ("A clerk files documents.", "clerk"),
        ("The accountant manages finances.", "accountant"),
        ("A banker handles transactions.", "banker"),
        ("The economist studies markets.", "economist"),
        ("A merchant sells goods.", "merchant"),
        ("The cashier processes payments.", "cashier"),
        ("A salesman pitches products.", "salesman"),
        ("The manager supervises teams.", "manager"),
        ("An executive makes decisions.", "executive"),
        ("The entrepreneur starts businesses.", "entrepreneur"),
        ("A consultant provides advice.", "consultant"),
        ("The coach trains athletes.", "coach"),
        ("An instructor teaches classes.", "instructor"),
        ("The trainer guides workouts.", "trainer"),
        ("A guide leads tours.", "guide"),
        ("The translator converts languages.", "translator"),
        ("An interpreter facilitates communication.", "interpreter"),
        ("The secretary schedules appointments.", "secretary"),
        ("An assistant helps with tasks.", "assistant"),
        ("The receptionist greets visitors.", "receptionist"),
        ("A custodian maintains facilities.", "custodian"),
        ("The janitor cleans buildings.", "janitor"),
        ("A guard watches premises.", "guard"),
        ("The volunteer contributes time.", "volunteer"),
        ("An intern learns skills.", "intern"),
        ("The apprentice studies a trade.", "apprentice"),
        ("A craftsman creates items.", "craftsman"),
        ("The tailor sews garments.", "tailor"),
        ("A barber cuts hair.", "barber"),
        ("The hairdresser styles hair.", "hairdresser"),
        ("A beautician applies makeup.", "beautician"),
        ("The optician fits glasses.", "optician"),
        ("A veterinarian treats animals.", "veterinarian"),
        ("The pharmacist dispenses medicine.", "pharmacist"),
        ("A chemist analyzes substances.", "chemist"),
        ("The biologist studies life.", "biologist"),
        ("A physicist explores matter.", "physicist"),
        ("The mathematician solves equations.", "mathematician"),
        ("An astronomer observes stars.", "astronomer"),
        ("The geologist examines rocks.", "geologist"),
        ("A meteorologist predicts weather.", "meteorologist"),
        ("The archaeologist excavates sites.", "archaeologist"),
        ("An anthropologist studies cultures.", "anthropologist"),
        ("The historian documents events.", "historian"),
        ("A philosopher ponders existence.", "philosopher"),

        # Objects and Things (200 examples)
        ("The book contains valuable information.", "book"),
        ("A chair supports people comfortably.", "chair"),
        ("The table holds various items.", "table"),
        ("A desk provides workspace.", "desk"),
        ("The lamp illuminates the room.", "lamp"),
        ("A candle flickers gently.", "candle"),
        ("The door opens inward.", "door"),
        ("A window provides ventilation.", "window"),
        ("The wall stands firmly.", "wall"),
        ("A floor needs cleaning.", "floor"),
        ("The ceiling has been painted.", "ceiling"),
        ("A roof protects from rain.", "roof"),
        ("The house looks welcoming.", "house"),
        ("A building towers impressively.", "building"),
        ("The bridge spans the river.", "bridge"),
        ("A road leads somewhere.", "road"),
        ("The path winds through trees.", "path"),
        ("A street bustles with activity.", "street"),
        ("The car drives smoothly.", "car"),
        ("A truck carries cargo.", "truck"),
        ("The bus transports passengers.", "bus"),
        ("A train arrives punctually.", "train"),
        ("The bicycle leans against the wall.", "bicycle"),
        ("A motorcycle roars loudly.", "motorcycle"),
        ("The airplane flies overhead.", "airplane"),
        ("A helicopter hovers nearby.", "helicopter"),
        ("The boat floats peacefully.", "boat"),
        ("A ship sails the ocean.", "ship"),
        ("The computer processes information.", "computer"),
        ("A phone rings insistently.", "phone"),
        ("The tablet displays content.", "tablet"),
        ("A laptop runs efficiently.", "laptop"),
        ("The keyboard clicks rhythmically.", "keyboard"),
        ("A mouse moves precisely.", "mouse"),
        ("The screen shows results.", "screen"),
        ("A monitor displays graphics.", "monitor"),
        ("The printer produces documents.", "printer"),
        ("A scanner digitizes images.", "scanner"),
        ("The camera captures moments.", "camera"),
        ("A microphone records audio.", "microphone"),
        ("The speaker plays music.", "speaker"),
        ("A headphone delivers sound.", "headphone"),
        ("The television broadcasts programs.", "television"),
        ("A radio receives signals.", "radio"),
        ("The refrigerator keeps food fresh.", "refrigerator"),
        ("An oven bakes food.", "oven"),
        ("The stove heats pots.", "stove"),
        ("A microwave warms meals.", "microwave"),
        ("The toaster browns bread.", "toaster"),
        ("A blender mixes ingredients.", "blender"),
        ("The dishwasher cleans plates.", "dishwasher"),
        ("A washer cleans clothes.", "washer"),
        ("The dryer removes moisture.", "dryer"),
        ("A vacuum removes dirt.", "vacuum"),
        ("The fan circulates air.", "fan"),
        ("An air-conditioner cools spaces.", "air-conditioner"),
        ("The heater warms rooms.", "heater"),
        ("A thermostat regulates temperature.", "thermostat"),
        ("The clock tells time.", "clock"),
        ("A watch shows hours.", "watch"),
        ("The calendar marks dates.", "calendar"),
        ("An alarm rings loudly.", "alarm"),
        ("The bell chimes melodiously.", "bell"),
        ("A whistle sounds sharply.", "whistle"),
        ("The siren wails urgently.", "siren"),
        ("A horn honks repeatedly.", "horn"),
        ("The pen writes smoothly.", "pen"),
        ("A pencil marks paper.", "pencil"),
        ("The marker draws boldly.", "marker"),
        ("A crayon colors brightly.", "crayon"),
        ("The brush paints surfaces.", "brush"),
        ("An eraser removes marks.", "eraser"),
        ("The ruler measures length.", "ruler"),
        ("A compass draws circles.", "compass"),
        ("The calculator computes numbers.", "calculator"),
        ("A notebook stores notes.", "notebook"),
        ("The journal records thoughts.", "journal"),
        ("A diary contains secrets.", "diary"),
        ("The magazine features articles.", "magazine"),
        ("A newspaper reports events.", "newspaper"),
        ("The novel tells stories.", "novel"),
        ("A textbook explains concepts.", "textbook"),
        ("The dictionary defines words.", "dictionary"),
        ("An encyclopedia provides knowledge.", "encyclopedia"),
        ("The atlas shows maps.", "atlas"),
        ("A manual gives instructions.", "manual"),
        ("The recipe describes cooking.", "recipe"),
        ("A map indicates locations.", "map"),
        ("The chart displays data.", "chart"),
        ("A graph visualizes trends.", "graph"),
        ("The diagram illustrates processes.", "diagram"),
        ("A picture shows scenes.", "picture"),
        ("The painting depicts beauty.", "painting"),
        ("A photograph freezes moments.", "photograph"),
        ("The sculpture stands prominently.", "sculpture"),
        ("A statue represents figures.", "statue"),
        ("The monument commemorates events.", "monument"),
        ("A trophy symbolizes achievement.", "trophy"),
        ("The medal honors excellence.", "medal"),
        ("A prize rewards winners.", "prize"),
        ("The certificate confirms completion.", "certificate"),
        ("A diploma proves graduation.", "diploma"),
        ("The license permits activities.", "license"),
        ("A passport enables travel.", "passport"),
        ("The ticket grants entry.", "ticket"),
        ("A receipt confirms payment.", "receipt"),
        ("The invoice requests payment.", "invoice"),
        ("A contract binds parties.", "contract"),
        ("The agreement establishes terms.", "agreement"),
        ("A document contains information.", "document"),
        ("The file organizes data.", "file"),
        ("A folder holds papers.", "folder"),
        ("The envelope contains letters.", "envelope"),
        ("A package arrives today.", "package"),
        ("The box stores items.", "box"),
        ("A container holds contents.", "container"),
        ("The basket carries goods.", "basket"),
        ("A bag holds belongings.", "bag"),
        ("The suitcase contains clothes.", "suitcase"),
        ("A backpack carries supplies.", "backpack"),
        ("The purse holds essentials.", "purse"),
        ("A wallet stores money.", "wallet"),
        ("The coin has value.", "coin"),
        ("A bill represents currency.", "bill"),
        ("The card enables transactions.", "card"),
        ("A key unlocks doors.", "key"),
        ("The lock secures entrances.", "lock"),
        ("A chain connects links.", "chain"),
        ("The rope ties objects.", "rope"),
        ("A string binds packages.", "string"),
        ("The wire conducts electricity.", "wire"),
        ("A cable transmits signals.", "cable"),
        ("The pipe carries water.", "pipe"),
        ("A hose sprays liquid.", "hose"),
        ("The tube contains paste.", "tube"),
        ("A bottle holds beverages.", "bottle"),
        ("The jar preserves food.", "jar"),
        ("A can stores goods.", "can"),
        ("The cup holds drinks.", "cup"),
        ("A glass contains liquid.", "glass"),
        ("The mug keeps coffee warm.", "mug"),
        ("A bowl contains soup.", "bowl"),
        ("The plate holds food.", "plate"),
        ("A dish serves meals.", "dish"),
        ("The spoon stirs ingredients.", "spoon"),
        ("A fork picks food.", "fork"),
        ("The knife cuts precisely.", "knife"),
        ("A chopstick helps eating.", "chopstick"),
        ("The napkin wipes hands.", "napkin"),
        ("A towel dries surfaces.", "towel"),
        ("The blanket provides warmth.", "blanket"),
        ("A pillow supports heads.", "pillow"),
        ("The mattress offers comfort.", "mattress"),
        ("A bed facilitates sleep.", "bed"),
        ("The couch seats people.", "couch"),
        ("A sofa provides seating.", "sofa"),
        ("The bench offers rest.", "bench"),
        ("A stool elevates height.", "stool"),

        # Nature and Places (150 examples)
        ("The mountain rises majestically.", "mountain"),
        ("A hill slopes gently.", "hill"),
        ("The valley stretches wide.", "valley"),
        ("A canyon cuts deep.", "canyon"),
        ("The cliff drops sharply.", "cliff"),
        ("A cave provides shelter.", "cave"),
        ("The river flows steadily.", "river"),
        ("A stream babbles softly.", "stream"),
        ("The lake reflects sky.", "lake"),
        ("A pond teems with life.", "pond"),
        ("The ocean crashes powerfully.", "ocean"),
        ("A sea extends endlessly.", "sea"),
        ("The wave crashes ashore.", "wave"),
        ("A tide rises predictably.", "tide"),
        ("The beach attracts visitors.", "beach"),
        ("A shore meets water.", "shore"),
        ("The island stands isolated.", "island"),
        ("A peninsula juts outward.", "peninsula"),
        ("The forest grows densely.", "forest"),
        ("A jungle thrives tropically.", "jungle"),
        ("The tree provides oxygen.", "tree"),
        ("A bush grows thickly.", "bush"),
        ("The shrub needs trimming.", "shrub"),
        ("A plant photosynthesizes daily.", "plant"),
        ("The flower blooms beautifully.", "flower"),
        ("A seed germinates slowly.", "seed"),
        ("The leaf changes color.", "leaf"),
        ("A branch extends outward.", "branch"),
        ("The trunk supports limbs.", "trunk"),
        ("A root anchors firmly.", "root"),
        ("The grass grows quickly.", "grass"),
        ("A weed spreads rapidly.", "weed"),
        ("The field yields crops.", "field"),
        ("A meadow blooms colorfully.", "meadow"),
        ("The prairie stretches far.", "prairie"),
        ("A plain extends flatly.", "plain"),
        ("The desert lacks water.", "desert"),
        ("A dune shifts constantly.", "dune"),
        ("The oasis offers refuge.", "oasis"),
        ("A tundra remains frozen.", "tundra"),
        ("The glacier moves slowly.", "glacier"),
        ("An iceberg floats dangerously.", "iceberg"),
        ("The snow falls gently.", "snow"),
        ("A snowflake drifts down.", "snowflake"),
        ("The ice forms overnight.", "ice"),
        ("A cloud drifts lazily.", "cloud"),
        ("The sky appears blue.", "sky"),
        ("A star twinkles brightly.", "star"),
        ("The sun shines warmly.", "sun"),
        ("A moon illuminates night.", "moon"),
        ("The planet orbits regularly.", "planet"),
        ("A meteor streaks across.", "meteor"),
        ("The comet appears rarely.", "comet"),
        ("A constellation forms patterns.", "constellation"),
        ("The galaxy contains billions.", "galaxy"),
        ("A universe expands infinitely.", "universe"),
        ("The atmosphere protects Earth.", "atmosphere"),
        ("A wind blows strongly.", "wind"),
        ("The breeze feels refreshing.", "breeze"),
        ("A storm approaches quickly.", "storm"),
        ("The rain falls steadily.", "rain"),
        ("A raindrop splashes down.", "raindrop"),
        ("The thunder rumbles loudly.", "thunder"),
        ("A lightning strikes suddenly.", "lightning"),
        ("The fog obscures vision.", "fog"),
        ("A mist rises gently.", "mist"),
        ("The dew forms overnight.", "dew"),
        ("A frost covers surfaces.", "frost"),
        ("The rainbow arcs beautifully.", "rainbow"),
        ("A tornado spins violently.", "tornado"),
        ("The hurricane devastates areas.", "hurricane"),
        ("A cyclone rotates powerfully.", "cyclone"),
        ("The earthquake shakes ground.", "earthquake"),
        ("A volcano erupts violently.", "volcano"),
        ("The lava flows hot.", "lava"),
        ("A rock sits motionless.", "rock"),
        ("The stone remains solid.", "stone"),
        ("A pebble skips smoothly.", "pebble"),
        ("The boulder blocks paths.", "boulder"),
        ("A mineral contains elements.", "mineral"),
        ("The crystal reflects light.", "crystal"),
        ("A gem sparkles brilliantly.", "gem"),
        ("The diamond shines forever.", "diamond"),
        ("A pearl forms naturally.", "pearl"),
        ("The gold gleams richly.", "gold"),
        ("A silver tarnishes slowly.", "silver"),
        ("The copper conducts well.", "copper"),
        ("An iron rusts easily.", "iron"),
        ("The steel remains strong.", "steel"),
        ("A metal conducts heat.", "metal"),
        ("The wood burns slowly.", "wood"),
        ("A log fuels fires.", "log"),
        ("The coal provides energy.", "coal"),
        ("An oil lubricates machinery.", "oil"),
        ("The gas expands freely.", "gas"),
        ("A liquid flows easily.", "liquid"),
        ("The water hydrates bodies.", "water"),
        ("A drop falls softly.", "drop"),
        ("The fire burns brightly.", "fire"),
        ("A flame flickers constantly.", "flame"),
        ("The smoke rises upward.", "smoke"),
        ("An ash settles down.", "ash"),
        ("The ember glows dimly.", "ember"),
        ("A spark ignites fuel.", "spark"),
        ("The explosion occurs suddenly.", "explosion"),
        ("A sound travels through air.", "sound"),
        ("The noise disturbs peace.", "noise"),
        ("A silence feels peaceful.", "silence"),
        ("The echo repeats back.", "echo"),
        ("A voice speaks clearly.", "voice"),
        ("The tone conveys emotion.", "tone"),
        ("A pitch varies widely.", "pitch"),
        ("The rhythm beats steadily.", "rhythm"),
        ("A melody sounds pleasant.", "melody"),
        ("The harmony blends perfectly.", "harmony"),
        ("A chord resonates deeply.", "chord"),
        ("The note holds long.", "note"),
        ("A beat pulses regularly.", "beat"),
        ("The tempo changes frequently.", "tempo"),
        ("A song plays repeatedly.", "song"),
        ("The music fills space.", "music"),
        ("A tune sticks mentally.", "tune"),
        ("The symphony performs magnificently.", "symphony"),
        ("An orchestra plays together.", "orchestra"),
        ("The band performs live.", "band"),
        ("A choir sings harmoniously.", "choir"),
        ("The audience applauds loudly.", "audience"),
        ("A crowd gathers quickly.", "crowd"),
        ("The group meets regularly.", "group"),
        ("A team works together.", "team"),
        ("The club welcomes members.", "club"),
        ("An organization serves communities.", "organization"),
        ("The company employs workers.", "company"),
        ("A business operates daily.", "business"),
        ("The store sells products.", "store"),
        ("A shop offers services.", "shop"),
        ("The market bustles actively.", "market"),
        ("A mall attracts shoppers.", "mall"),
        ("The restaurant serves meals.", "restaurant"),
        ("A cafe offers drinks.", "cafe"),
        ("The bar serves beverages.", "bar"),
        ("A hotel accommodates guests.", "hotel"),
        ("The hospital treats patients.", "hospital"),
        ("A clinic provides care.", "clinic"),
        ("The school educates students.", "school"),
        ("A university offers degrees.", "university"),
        ("The college prepares graduates.", "college"),
        ("A library lends books.", "library"),
        ("The museum displays artifacts.", "museum"),
        ("A gallery exhibits art.", "gallery"),
        ("The theater shows performances.", "theater"),
        ("A cinema screens films.", "cinema"),
    ]

    # Generate 500 unique plural examples (matching singular examples)
    plural_examples = [
        # Animals (50 examples)
        ("The cats sit on the windowsill.", "cats"),
        ("Dogs bark at strangers.", "Dogs"),
        ("The birds sing in the morning.", "birds"),
        ("Horses gallop across the field.", "Horses"),
        ("The rabbits hop through the garden.", "rabbits"),
        ("Lions roar in the jungle.", "Lions"),
        ("The elephants walk slowly.", "elephants"),
        ("Tigers hunt at night.", "Tigers"),
        ("The monkeys swing from trees.", "monkeys"),
        ("Dolphins swim gracefully.", "Dolphins"),
        ("The penguins waddle on ice.", "penguins"),
        ("Bears hibernate in winter.", "Bears"),
        ("The snakes slither quietly.", "snakes"),
        ("Frogs jump into the pond.", "Frogs"),
        ("The butterflies emerge from their cocoons.", "butterflies"),
        ("Bees buzz around flowers.", "Bees"),
        ("The ants carry heavy loads.", "ants"),
        ("Spiders spin their webs.", "Spiders"),
        ("The fish swim upstream.", "fish"),
        ("Sharks patrol the waters.", "Sharks"),
        ("The whales breach the surface.", "whales"),
        ("Eagles soar above mountains.", "Eagles"),
        ("The owls hoot at midnight.", "owls"),
        ("Parrots repeat words.", "Parrots"),
        ("The crows caw loudly.", "crows"),
        ("Seagulls fly over the ocean.", "Seagulls"),
        ("The ducks quack in the pond.", "ducks"),
        ("Geese honk aggressively.", "Geese"),
        ("The turkeys gobble nervously.", "turkeys"),
        ("Chickens peck at grain.", "Chickens"),
        ("The cows moo in the barn.", "cows"),
        ("Pigs wallow in mud.", "Pigs"),
        ("The sheep graze on grass.", "sheep"),
        ("Goats climb the rocks.", "Goats"),
        ("The donkeys bray stubbornly.", "donkeys"),
        ("Camels travel through deserts.", "Camels"),
        ("The giraffes reach high branches.", "giraffes"),
        ("Zebras have distinctive stripes.", "Zebras"),
        ("The rhinos charge forward.", "rhinos"),
        ("Hippos rest in water.", "Hippos"),
        ("The kangaroos jump far.", "kangaroos"),
        ("Koalas sleep in trees.", "Koalas"),
        ("The pandas eat bamboo.", "pandas"),
        ("Wolves howl at the moon.", "Wolves"),
        ("The foxes hunt cleverly.", "foxes"),
        ("Deer run through the forest.", "Deer"),
        ("The moose have large antlers.", "moose"),
        ("Raccoons search for food.", "Raccoons"),
        ("The squirrels collect nuts.", "squirrels"),
        ("Mice scurry away.", "Mice"),

        # People and Professions (100 examples)
        ("The students study diligently.", "students"),
        ("Teachers explain concepts clearly.", "Teachers"),
        ("The professors lecture enthusiastically.", "professors"),
        ("Doctors examine patients carefully.", "Doctors"),
        ("The nurses administer medication.", "nurses"),
        ("Surgeons perform operations.", "Surgeons"),
        ("The dentists clean teeth.", "dentists"),
        ("Therapists listen attentively.", "Therapists"),
        ("The scientists conduct experiments.", "scientists"),
        ("Researchers analyze data.", "Researchers"),
        ("The engineers design systems.", "engineers"),
        ("Programmers write code.", "Programmers"),
        ("The developers build applications.", "developers"),
        ("Designers create graphics.", "Designers"),
        ("The artists paint masterpieces.", "artists"),
        ("Musicians play instruments.", "Musicians"),
        ("The singers perform songs.", "singers"),
        ("Dancers move gracefully.", "Dancers"),
        ("The actors portray characters.", "actors"),
        ("Directors manage productions.", "Directors"),
        ("The writers compose stories.", "writers"),
        ("Authors publish books.", "Authors"),
        ("The poets craft verses.", "poets"),
        ("Journalists report news.", "Journalists"),
        ("The editors review manuscripts.", "editors"),
        ("Photographers capture images.", "Photographers"),
        ("The chefs prepare meals.", "chefs"),
        ("Cooks follow recipes.", "Cooks"),
        ("The waiters serve customers.", "waiters"),
        ("Bartenders mix drinks.", "Bartenders"),
        ("The lawyers argue cases.", "lawyers"),
        ("Attorneys represent clients.", "Attorneys"),
        ("The judges preside over trials.", "judges"),
        ("Politicians campaign actively.", "Politicians"),
        ("The mayors govern the cities.", "mayors"),
        ("Senators propose legislation.", "Senators"),
        ("The officers patrol the streets.", "officers"),
        ("Detectives solve crimes.", "Detectives"),
        ("The firefighters rescue people.", "firefighters"),
        ("Paramedics provide emergency care.", "Paramedics"),
        ("The soldiers follow orders.", "soldiers"),
        ("Pilots fly aircraft.", "Pilots"),
        ("The captains command the ships.", "captains"),
        ("Astronauts explore space.", "Astronauts"),
        ("The mechanics repair vehicles.", "mechanics"),
        ("Plumbers fix pipes.", "Plumbers"),
        ("The electricians install wiring.", "electricians"),
        ("Carpenters build furniture.", "Carpenters"),
        ("The architects plan buildings.", "architects"),
        ("Contractors manage construction.", "Contractors"),
        ("The farmers grow crops.", "farmers"),
        ("Gardeners tend plants.", "Gardeners"),
        ("The fishermen catch fish.", "fishermen"),
        ("Miners extract minerals.", "Miners"),
        ("The librarians organize books.", "librarians"),
        ("Clerks file documents.", "Clerks"),
        ("The accountants manage finances.", "accountants"),
        ("Bankers handle transactions.", "Bankers"),
        ("The economists study markets.", "economists"),
        ("Merchants sell goods.", "Merchants"),
        ("The cashiers process payments.", "cashiers"),
        ("Salesmen pitch products.", "Salesmen"),
        ("The managers supervise teams.", "managers"),
        ("Executives make decisions.", "Executives"),
        ("The entrepreneurs start businesses.", "entrepreneurs"),
        ("Consultants provide advice.", "Consultants"),
        ("The coaches train athletes.", "coaches"),
        ("Instructors teach classes.", "Instructors"),
        ("The trainers guide workouts.", "trainers"),
        ("Guides lead tours.", "Guides"),
        ("The translators convert languages.", "translators"),
        ("Interpreters facilitate communication.", "Interpreters"),
        ("The secretaries schedule appointments.", "secretaries"),
        ("Assistants help with tasks.", "Assistants"),
        ("The receptionists greet visitors.", "receptionists"),
        ("Custodians maintain facilities.", "Custodians"),
        ("The janitors clean buildings.", "janitors"),
        ("Guards watch premises.", "Guards"),
        ("The volunteers contribute time.", "volunteers"),
        ("Interns learn skills.", "Interns"),
        ("The apprentices study trades.", "apprentices"),
        ("Craftsmen create items.", "Craftsmen"),
        ("The tailors sew garments.", "tailors"),
        ("Barbers cut hair.", "Barbers"),
        ("The hairdressers style hair.", "hairdressers"),
        ("Beauticians apply makeup.", "Beauticians"),
        ("The opticians fit glasses.", "opticians"),
        ("Veterinarians treat animals.", "Veterinarians"),
        ("The pharmacists dispense medicine.", "pharmacists"),
        ("Chemists analyze substances.", "Chemists"),
        ("The biologists study life.", "biologists"),
        ("Physicists explore matter.", "Physicists"),
        ("The mathematicians solve equations.", "mathematicians"),
        ("Astronomers observe stars.", "Astronomers"),
        ("The geologists examine rocks.", "geologists"),
        ("Meteorologists predict weather.", "Meteorologists"),
        ("The archaeologists excavate sites.", "archaeologists"),
        ("Anthropologists study cultures.", "Anthropologists"),
        ("The historians document events.", "historians"),
        ("Philosophers ponder existence.", "Philosophers"),

        # Objects and Things (200 examples)
        ("The books contain valuable information.", "books"),
        ("Chairs support people comfortably.", "Chairs"),
        ("The tables hold various items.", "tables"),
        ("Desks provide workspace.", "Desks"),
        ("The lamps illuminate the room.", "lamps"),
        ("Candles flicker gently.", "Candles"),
        ("The doors open inward.", "doors"),
        ("Windows provide ventilation.", "Windows"),
        ("The walls stand firmly.", "walls"),
        ("Floors need cleaning.", "Floors"),
        ("The ceilings have been painted.", "ceilings"),
        ("Roofs protect from rain.", "Roofs"),
        ("The houses look welcoming.", "houses"),
        ("Buildings tower impressively.", "Buildings"),
        ("The bridges span the rivers.", "bridges"),
        ("Roads lead somewhere.", "Roads"),
        ("The paths wind through trees.", "paths"),
        ("Streets bustle with activity.", "Streets"),
        ("The cars drive smoothly.", "cars"),
        ("Trucks carry cargo.", "Trucks"),
        ("The buses transport passengers.", "buses"),
        ("Trains arrive punctually.", "Trains"),
        ("The bicycles lean against the wall.", "bicycles"),
        ("Motorcycles roar loudly.", "Motorcycles"),
        ("The airplanes fly overhead.", "airplanes"),
        ("Helicopters hover nearby.", "Helicopters"),
        ("The boats float peacefully.", "boats"),
        ("Ships sail the ocean.", "Ships"),
        ("The computers process information.", "computers"),
        ("Phones ring insistently.", "Phones"),
        ("The tablets display content.", "tablets"),
        ("Laptops run efficiently.", "Laptops"),
        ("The keyboards click rhythmically.", "keyboards"),
        ("Mice move precisely.", "Mice"),
        ("The screens show results.", "screens"),
        ("Monitors display graphics.", "Monitors"),
        ("The printers produce documents.", "printers"),
        ("Scanners digitize images.", "Scanners"),
        ("The cameras capture moments.", "cameras"),
        ("Microphones record audio.", "Microphones"),
        ("The speakers play music.", "speakers"),
        ("Headphones deliver sound.", "Headphones"),
        ("The televisions broadcast programs.", "televisions"),
        ("Radios receive signals.", "Radios"),
        ("The refrigerators keep food fresh.", "refrigerators"),
        ("Ovens bake food.", "Ovens"),
        ("The stoves heat pots.", "stoves"),
        ("Microwaves warm meals.", "Microwaves"),
        ("The toasters brown bread.", "toasters"),
        ("Blenders mix ingredients.", "Blenders"),
        ("The dishwashers clean plates.", "dishwashers"),
        ("Washers clean clothes.", "Washers"),
        ("The dryers remove moisture.", "dryers"),
        ("Vacuums remove dirt.", "Vacuums"),
        ("The fans circulate air.", "fans"),
        ("Air-conditioners cool spaces.", "Air-conditioners"),
        ("The heaters warm rooms.", "heaters"),
        ("Thermostats regulate temperature.", "Thermostats"),
        ("The clocks tell time.", "clocks"),
        ("Watches show hours.", "Watches"),
        ("The calendars mark dates.", "calendars"),
        ("Alarms ring loudly.", "Alarms"),
        ("The bells chime melodiously.", "bells"),
        ("Whistles sound sharply.", "Whistles"),
        ("The sirens wail urgently.", "sirens"),
        ("Horns honk repeatedly.", "Horns"),
        ("The pens write smoothly.", "pens"),
        ("Pencils mark paper.", "Pencils"),
        ("The markers draw boldly.", "markers"),
        ("Crayons color brightly.", "Crayons"),
        ("The brushes paint surfaces.", "brushes"),
        ("Erasers remove marks.", "Erasers"),
        ("The rulers measure length.", "rulers"),
        ("Compasses draw circles.", "Compasses"),
        ("The calculators compute numbers.", "calculators"),
        ("Notebooks store notes.", "Notebooks"),
        ("The journals record thoughts.", "journals"),
        ("Diaries contain secrets.", "Diaries"),
        ("The magazines feature articles.", "magazines"),
        ("Newspapers report events.", "Newspapers"),
        ("The novels tell stories.", "novels"),
        ("Textbooks explain concepts.", "Textbooks"),
        ("The dictionaries define words.", "dictionaries"),
        ("Encyclopedias provide knowledge.", "Encyclopedias"),
        ("The atlases show maps.", "atlases"),
        ("Manuals give instructions.", "Manuals"),
        ("The recipes describe cooking.", "recipes"),
        ("Maps indicate locations.", "Maps"),
        ("The charts display data.", "charts"),
        ("Graphs visualize trends.", "Graphs"),
        ("The diagrams illustrate processes.", "diagrams"),
        ("Pictures show scenes.", "Pictures"),
        ("The paintings depict beauty.", "paintings"),
        ("Photographs freeze moments.", "Photographs"),
        ("The sculptures stand prominently.", "sculptures"),
        ("Statues represent figures.", "Statues"),
        ("The monuments commemorate events.", "monuments"),
        ("Trophies symbolize achievement.", "Trophies"),
        ("The medals honor excellence.", "medals"),
        ("Prizes reward winners.", "Prizes"),
        ("The certificates confirm completion.", "certificates"),
        ("Diplomas prove graduation.", "Diplomas"),
        ("The licenses permit activities.", "licenses"),
        ("Passports enable travel.", "Passports"),
        ("The tickets grant entry.", "tickets"),
        ("Receipts confirm payment.", "Receipts"),
        ("The invoices request payment.", "invoices"),
        ("Contracts bind parties.", "Contracts"),
        ("The agreements establish terms.", "agreements"),
        ("Documents contain information.", "Documents"),
        ("The files organize data.", "files"),
        ("Folders hold papers.", "Folders"),
        ("The envelopes contain letters.", "envelopes"),
        ("Packages arrive today.", "Packages"),
        ("The boxes store items.", "boxes"),
        ("Containers hold contents.", "Containers"),
        ("The baskets carry goods.", "baskets"),
        ("Bags hold belongings.", "Bags"),
        ("The suitcases contain clothes.", "suitcases"),
        ("Backpacks carry supplies.", "Backpacks"),
        ("The purses hold essentials.", "purses"),
        ("Wallets store money.", "Wallets"),
        ("The coins have value.", "coins"),
        ("Bills represent currency.", "Bills"),
        ("The cards enable transactions.", "cards"),
        ("Keys unlock doors.", "Keys"),
        ("The locks secure entrances.", "locks"),
        ("Chains connect links.", "Chains"),
        ("The ropes tie objects.", "ropes"),
        ("Strings bind packages.", "Strings"),
        ("The wires conduct electricity.", "wires"),
        ("Cables transmit signals.", "Cables"),
        ("The pipes carry water.", "pipes"),
        ("Hoses spray liquid.", "Hoses"),
        ("The tubes contain paste.", "tubes"),
        ("Bottles hold beverages.", "Bottles"),
        ("The jars preserve food.", "jars"),
        ("Cans store goods.", "Cans"),
        ("The cups hold drinks.", "cups"),
        ("Glasses contain liquid.", "Glasses"),
        ("The mugs keep coffee warm.", "mugs"),
        ("Bowls contain soup.", "Bowls"),
        ("The plates hold food.", "plates"),
        ("Dishes serve meals.", "Dishes"),
        ("The spoons stir ingredients.", "spoons"),
        ("Forks pick food.", "Forks"),
        ("The knives cut precisely.", "knives"),
        ("Chopsticks help eating.", "Chopsticks"),
        ("The napkins wipe hands.", "napkins"),
        ("Towels dry surfaces.", "Towels"),
        ("The blankets provide warmth.", "blankets"),
        ("Pillows support heads.", "Pillows"),
        ("The mattresses offer comfort.", "mattresses"),
        ("Beds facilitate sleep.", "Beds"),
        ("The couches seat people.", "couches"),
        ("Sofas provide seating.", "Sofas"),
        ("The benches offer rest.", "benches"),
        ("Stools elevate height.", "Stools"),

        # Nature and Places (150 examples)
        ("The mountains rise majestically.", "mountains"),
        ("Hills slope gently.", "Hills"),
        ("The valleys stretch wide.", "valleys"),
        ("Canyons cut deep.", "Canyons"),
        ("The cliffs drop sharply.", "cliffs"),
        ("Caves provide shelter.", "Caves"),
        ("The rivers flow steadily.", "rivers"),
        ("Streams babble softly.", "Streams"),
        ("The lakes reflect sky.", "lakes"),
        ("Ponds teem with life.", "Ponds"),
        ("The oceans crash powerfully.", "oceans"),
        ("Seas extend endlessly.", "Seas"),
        ("The waves crash ashore.", "waves"),
        ("Tides rise predictably.", "Tides"),
        ("The beaches attract visitors.", "beaches"),
        ("Shores meet water.", "Shores"),
        ("The islands stand isolated.", "islands"),
        ("Peninsulas jut outward.", "Peninsulas"),
        ("The forests grow densely.", "forests"),
        ("Jungles thrive tropically.", "Jungles"),
        ("The trees provide oxygen.", "trees"),
        ("Bushes grow thickly.", "Bushes"),
        ("The shrubs need trimming.", "shrubs"),
        ("Plants photosynthesize daily.", "Plants"),
        ("The flowers bloom beautifully.", "flowers"),
        ("Seeds germinate slowly.", "Seeds"),
        ("The leaves change color.", "leaves"),
        ("Branches extend outward.", "Branches"),
        ("The trunks support limbs.", "trunks"),
        ("Roots anchor firmly.", "Roots"),
        ("The grasses grow quickly.", "grasses"),
        ("Weeds spread rapidly.", "Weeds"),
        ("The fields yield crops.", "fields"),
        ("Meadows bloom colorfully.", "Meadows"),
        ("The prairies stretch far.", "prairies"),
        ("Plains extend flatly.", "Plains"),
        ("The deserts lack water.", "deserts"),
        ("Dunes shift constantly.", "Dunes"),
        ("The oases offer refuge.", "oases"),
        ("Tundras remain frozen.", "Tundras"),
        ("The glaciers move slowly.", "glaciers"),
        ("Icebergs float dangerously.", "Icebergs"),
        ("The snows fall gently.", "snows"),
        ("Snowflakes drift down.", "Snowflakes"),
        ("The ices form overnight.", "ices"),
        ("Clouds drift lazily.", "Clouds"),
        ("The skies appear blue.", "skies"),
        ("Stars twinkle brightly.", "Stars"),
        ("The suns shine warmly.", "suns"),
        ("Moons illuminate night.", "Moons"),
        ("The planets orbit regularly.", "planets"),
        ("Meteors streak across.", "Meteors"),
        ("The comets appear rarely.", "comets"),
        ("Constellations form patterns.", "Constellations"),
        ("The galaxies contain billions.", "galaxies"),
        ("Universes expand infinitely.", "Universes"),
        ("The atmospheres protect Earth.", "atmospheres"),
        ("Winds blow strongly.", "Winds"),
        ("The breezes feel refreshing.", "breezes"),
        ("Storms approach quickly.", "Storms"),
        ("The rains fall steadily.", "rains"),
        ("Raindrops splash down.", "Raindrops"),
        ("The thunders rumble loudly.", "thunders"),
        ("Lightnings strike suddenly.", "Lightnings"),
        ("The fogs obscure vision.", "fogs"),
        ("Mists rise gently.", "Mists"),
        ("The dews form overnight.", "dews"),
        ("Frosts cover surfaces.", "Frosts"),
        ("The rainbows arc beautifully.", "rainbows"),
        ("Tornadoes spin violently.", "Tornadoes"),
        ("The hurricanes devastate areas.", "hurricanes"),
        ("Cyclones rotate powerfully.", "Cyclones"),
        ("The earthquakes shake ground.", "earthquakes"),
        ("Volcanoes erupt violently.", "Volcanoes"),
        ("The lavas flow hot.", "lavas"),
        ("Rocks sit motionless.", "Rocks"),
        ("The stones remain solid.", "stones"),
        ("Pebbles skip smoothly.", "Pebbles"),
        ("The boulders block paths.", "boulders"),
        ("Minerals contain elements.", "Minerals"),
        ("The crystals reflect light.", "crystals"),
        ("Gems sparkle brilliantly.", "Gems"),
        ("The diamonds shine forever.", "diamonds"),
        ("Pearls form naturally.", "Pearls"),
        ("The golds gleam richly.", "golds"),
        ("Silvers tarnish slowly.", "Silvers"),
        ("The coppers conduct well.", "coppers"),
        ("Irons rust easily.", "Irons"),
        ("The steels remain strong.", "steels"),
        ("Metals conduct heat.", "Metals"),
        ("The woods burn slowly.", "woods"),
        ("Logs fuel fires.", "Logs"),
        ("The coals provide energy.", "coals"),
        ("Oils lubricate machinery.", "Oils"),
        ("The gases expand freely.", "gases"),
        ("Liquids flow easily.", "Liquids"),
        ("The waters hydrate bodies.", "waters"),
        ("Drops fall softly.", "Drops"),
        ("The fires burn brightly.", "fires"),
        ("Flames flicker constantly.", "Flames"),
        ("The smokes rise upward.", "smokes"),
        ("Ashes settle down.", "Ashes"),
        ("The embers glow dimly.", "embers"),
        ("Sparks ignite fuel.", "Sparks"),
        ("The explosions occur suddenly.", "explosions"),
        ("Sounds travel through air.", "Sounds"),
        ("The noises disturb peace.", "noises"),
        ("Silences feel peaceful.", "Silences"),
        ("The echoes repeat back.", "echoes"),
        ("Voices speak clearly.", "Voices"),
        ("The tones convey emotion.", "tones"),
        ("Pitches vary widely.", "Pitches"),
        ("The rhythms beat steadily.", "rhythms"),
        ("Melodies sound pleasant.", "Melodies"),
        ("The harmonies blend perfectly.", "harmonies"),
        ("Chords resonate deeply.", "Chords"),
        ("The notes hold long.", "notes"),
        ("Beats pulse regularly.", "Beats"),
        ("The tempos change frequently.", "tempos"),
        ("Songs play repeatedly.", "Songs"),
        ("The musics fill space.", "musics"),
        ("Tunes stick mentally.", "Tunes"),
        ("The symphonies perform magnificently.", "symphonies"),
        ("Orchestras play together.", "Orchestras"),
        ("The bands perform live.", "bands"),
        ("Choirs sing harmoniously.", "Choirs"),
        ("The audiences applaud loudly.", "audiences"),
        ("Crowds gather quickly.", "Crowds"),
        ("The groups meet regularly.", "groups"),
        ("Teams work together.", "Teams"),
        ("The clubs welcome members.", "clubs"),
        ("Organizations serve communities.", "Organizations"),
        ("The companies employ workers.", "companies"),
        ("Businesses operate daily.", "Businesses"),
        ("The stores sell products.", "stores"),
        ("Shops offer services.", "Shops"),
        ("The markets bustle actively.", "markets"),
        ("Malls attract shoppers.", "Malls"),
        ("The restaurants serve meals.", "restaurants"),
        ("Cafes offer drinks.", "Cafes"),
        ("The bars serve beverages.", "bars"),
        ("Hotels accommodate guests.", "Hotels"),
        ("The hospitals treat patients.", "hospitals"),
        ("Clinics provide care.", "Clinics"),
        ("The schools educate students.", "schools"),
        ("Universities offer degrees.", "Universities"),
        ("The colleges prepare graduates.", "colleges"),
        ("Libraries lend books.", "Libraries"),
        ("The museums display artifacts.", "museums"),
        ("Galleries exhibit art.", "Galleries"),
        ("The theaters show performances.", "theaters"),
        ("Cinemas screen films.", "Cinemas"),
    ]

    dataset = []

    # Add all unique singular examples
    for text, target in singular_examples:
        dataset.append({'text': text, 'target_word': target, 'label': 0})

    # Add all unique plural examples
    for text, target in plural_examples:
        dataset.append({'text': text, 'target_word': target, 'label': 1})

    return dataset


def create_pos_dataset() -> List[Dict]:
    """
    Create dataset for part-of-speech prediction task.

    Returns 200 unique examples for each POS category (800 total).

    Returns:
        List of 800 examples with 'text', 'target_word', and 'label'
        (0=noun, 1=verb, 2=adjective, 3=adverb)
    """
    return generate_pos_dataset()


def create_verb_tense_dataset() -> List[Dict]:
    """
    Create dataset for verb tense prediction task.

    Returns 200 unique examples for each tense (600 total).

    Returns:
        List of 600 examples with 'text', 'target_word', and 'label'
        (0=past, 1=present, 2=future)
    """
    dataset = []

    # Past tense verbs (200 examples)
    past_verbs = [
        ("walked", "Yesterday I walked to the store."),
        ("ran", "She ran five miles last week."),
        ("ate", "They ate dinner at six."),
        ("wrote", "He wrote a long letter."),
        ("read", "I read that book yesterday."),
        ("spoke", "She spoke to the manager."),
        ("went", "They went home early."),
        ("came", "He came to visit us."),
        ("saw", "I saw a movie last night."),
        ("knew", "She knew the answer already."),
        ("thought", "He thought about it carefully."),
        ("felt", "They felt very happy then."),
        ("became", "She became a doctor."),
        ("left", "I left my keys there."),
        ("brought", "He brought flowers yesterday."),
        ("began", "The show began at eight."),
        ("kept", "She kept her promise."),
        ("held", "They held a meeting."),
        ("heard", "I heard a strange noise."),
        ("let", "She let me borrow it."),
        ("meant", "He meant what he said."),
        ("met", "They met at the cafe."),
        ("paid", "I paid the bill yesterday."),
        ("sat", "She sat by the window."),
        ("stood", "He stood in line."),
        ("understood", "They understood the problem."),
        ("won", "She won the competition."),
        ("lost", "He lost his wallet."),
        ("built", "They built a new house."),
        ("bought", "I bought fresh bread."),
        ("caught", "She caught the ball."),
        ("chose", "He chose the red one."),
        ("drew", "They drew beautiful pictures."),
        ("drove", "I drove to work."),
        ("fell", "She fell down the stairs."),
        ("flew", "The bird flew away."),
        ("forgot", "He forgot his password."),
        ("froze", "The lake froze over."),
        ("gave", "She gave me advice."),
        ("grew", "They grew vegetables."),
        ("hid", "I hid the present."),
        ("hit", "She hit the target."),
        ("hurt", "He hurt his ankle."),
        ("laid", "They laid the foundation."),
        ("led", "She led the team."),
        ("lent", "He lent me money."),
        ("lit", "I lit the candles."),
        ("made", "She made breakfast."),
        ("rode", "He rode his bicycle."),
        ("sang", "They sang together."),
        ("sent", "I sent an email."),
        ("shook", "She shook her head."),
        ("shot", "He shot the photo."),
        ("showed", "They showed the way."),
        ("shut", "I shut the door."),
        ("slept", "She slept well."),
        ("spent", "He spent all day."),
        ("split", "They split the cost."),
        ("spread", "I spread the butter."),
        ("stole", "Someone stole my bike."),
        ("struck", "Lightning struck twice."),
        ("swam", "She swam in the pool."),
        ("swore", "He swore to tell truth."),
        ("taught", "They taught the class."),
        ("threw", "I threw the ball."),
        ("told", "She told a story."),
        ("took", "He took the train."),
        ("tore", "They tore the paper."),
        ("woke", "I woke up early."),
        ("wore", "She wore a dress."),
        ("found", "He found the key."),
        ("sold", "They sold their car."),
        ("broke", "I broke the vase."),
        ("drank", "She drank some water."),
        ("fought", "They fought bravely."),
        ("got", "He got a promotion."),
        ("hung", "I hung the picture."),
        ("learned", "She learned quickly."),
        ("smelled", "It smelled delicious."),
        ("spun", "The wheel spun around."),
        ("stuck", "The door stuck."),
        ("swung", "The gate swung open."),
        ("worked", "I worked all day."),
        ("played", "She played piano."),
        ("studied", "He studied hard."),
        ("traveled", "They traveled abroad."),
        ("visited", "I visited my aunt."),
        ("watched", "She watched TV."),
        ("listened", "He listened carefully."),
        ("talked", "They talked for hours."),
        ("looked", "I looked everywhere."),
        ("seemed", "She seemed tired."),
        ("appeared", "He appeared suddenly."),
        ("happened", "It happened yesterday."),
        ("changed", "Things changed quickly."),
        ("moved", "I moved last year."),
        ("lived", "She lived in Paris."),
        ("died", "He died peacefully."),
        ("smiled", "They smiled warmly."),
        ("laughed", "I laughed loudly."),
        ("cried", "She cried sadly."),
        ("shouted", "He shouted angrily."),
        ("whispered", "They whispered quietly."),
        ("answered", "I answered correctly."),
        ("asked", "She asked politely."),
        ("called", "He called yesterday."),
        ("replied", "They replied promptly."),
        ("agreed", "I agreed completely."),
        ("argued", "She argued strongly."),
        ("believed", "He believed firmly."),
        ("decided", "They decided wisely."),
        ("explained", "I explained clearly."),
        ("helped", "She helped greatly."),
        ("hoped", "He hoped sincerely."),
        ("imagined", "They imagined vividly."),
        ("jumped", "I jumped high."),
        ("kicked", "She kicked hard."),
        ("kissed", "He kissed gently."),
        ("liked", "They liked it."),
        ("loved", "I loved deeply."),
        ("missed", "She missed badly."),
        ("needed", "He needed urgently."),
        ("noticed", "They noticed immediately."),
        ("offered", "I offered generously."),
        ("opened", "She opened carefully."),
        ("closed", "He closed tightly."),
        ("passed", "They passed successfully."),
        ("pulled", "I pulled strongly."),
        ("pushed", "She pushed hard."),
        ("reached", "He reached high."),
        ("received", "They received warmly."),
        ("remembered", "I remembered clearly."),
        ("repeated", "She repeated twice."),
        ("returned", "He returned safely."),
        ("saved", "They saved enough."),
        ("searched", "I searched thoroughly."),
        ("stayed", "She stayed overnight."),
        ("stopped", "He stopped suddenly."),
        ("tried", "They tried hard."),
        ("turned", "I turned around."),
        ("used", "She used it."),
        ("waited", "He waited patiently."),
        ("walked", "They walked slowly."),
        ("wanted", "I wanted more."),
        ("washed", "She washed carefully."),
        ("wished", "He wished hopefully."),
        ("wondered", "They wondered curiously."),
        ("worried", "I worried needlessly."),
        ("yelled", "She yelled loudly."),
        ("enjoyed", "He enjoyed thoroughly."),
        ("finished", "They finished early."),
        ("started", "I started immediately."),
        ("continued", "She continued bravely."),
        ("followed", "He followed closely."),
        ("guided", "They guided expertly."),
        ("joined", "I joined recently."),
        ("created", "She created beautifully."),
        ("destroyed", "He destroyed completely."),
        ("improved", "They improved significantly."),
        ("increased", "It increased rapidly."),
        ("decreased", "It decreased slowly."),
        ("developed", "She developed skillfully."),
        ("discovered", "He discovered accidentally."),
        ("invented", "They invented cleverly."),
        ("produced", "I produced efficiently."),
        ("protected", "She protected fiercely."),
        ("provided", "He provided generously."),
        ("raised", "They raised carefully."),
        ("reduced", "I reduced drastically."),
        ("removed", "She removed completely."),
        ("replaced", "He replaced quickly."),
        ("required", "They required immediately."),
        ("served", "I served faithfully."),
        ("shared", "She shared kindly."),
        ("suffered", "He suffered greatly."),
        ("suggested", "They suggested wisely."),
        ("supported", "I supported strongly."),
        ("supposed", "She supposed correctly."),
        ("surprised", "He surprised totally."),
        ("survived", "They survived miraculously."),
        ("touched", "I touched gently."),
        ("treated", "She treated fairly."),
        ("trusted", "He trusted completely."),
        ("valued", "They valued highly."),
        ("voted", "I voted yesterday."),
        ("warned", "She warned repeatedly."),
        ("welcomed", "He welcomed warmly."),
    ]

    for verb, text in past_verbs:
        dataset.append({
            'text': text,
            'target_word': verb,
            'label': 0
        })

    # Present tense verbs (200 examples)
    present_verbs = [
        ("walk", "I walk to school daily."),
        ("run", "She runs every morning."),
        ("eat", "They eat dinner together."),
        ("write", "He writes stories often."),
        ("read", "I read books regularly."),
        ("speak", "She speaks three languages."),
        ("go", "They go shopping weekly."),
        ("come", "He comes here frequently."),
        ("see", "I see her sometimes."),
        ("know", "She knows the truth."),
        ("think", "He thinks deeply."),
        ("feel", "They feel excited."),
        ("become", "She becomes stronger."),
        ("leave", "I leave at noon."),
        ("bring", "He brings lunch daily."),
        ("begin", "The show begins soon."),
        ("keep", "She keeps trying."),
        ("hold", "They hold meetings."),
        ("hear", "I hear music."),
        ("let", "She lets me help."),
        ("mean", "He means well."),
        ("meet", "They meet regularly."),
        ("pay", "I pay attention."),
        ("sit", "She sits here."),
        ("stand", "He stands tall."),
        ("understand", "They understand completely."),
        ("win", "She wins often."),
        ("lose", "He loses sometimes."),
        ("build", "They build houses."),
        ("buy", "I buy groceries."),
        ("catch", "She catches fish."),
        ("choose", "He chooses wisely."),
        ("draw", "They draw pictures."),
        ("drive", "I drive carefully."),
        ("fall", "She falls asleep."),
        ("fly", "Birds fly south."),
        ("forget", "He forgets things."),
        ("freeze", "Water freezes cold."),
        ("give", "She gives freely."),
        ("grow", "They grow plants."),
        ("hide", "I hide sometimes."),
        ("hit", "She hits targets."),
        ("hurt", "He hurts easily."),
        ("lay", "They lay foundations."),
        ("lead", "She leads effectively."),
        ("lend", "He lends money."),
        ("light", "I light candles."),
        ("make", "She makes art."),
        ("ride", "He rides horses."),
        ("sing", "They sing beautifully."),
        ("send", "I send messages."),
        ("shake", "She shakes hands."),
        ("shoot", "He shoots photos."),
        ("show", "They show kindness."),
        ("shut", "I shut doors."),
        ("sleep", "She sleeps soundly."),
        ("spend", "He spends wisely."),
        ("split", "They split evenly."),
        ("spread", "I spread joy."),
        ("steal", "Thieves steal."),
        ("strike", "Lightning strikes."),
        ("swim", "She swims fast."),
        ("swear", "He swears honestly."),
        ("teach", "They teach well."),
        ("throw", "I throw accurately."),
        ("tell", "She tells stories."),
        ("take", "He takes notes."),
        ("tear", "They tear easily."),
        ("wake", "I wake early."),
        ("wear", "She wears hats."),
        ("find", "He finds solutions."),
        ("sell", "They sell products."),
        ("break", "I break records."),
        ("drink", "She drinks tea."),
        ("fight", "They fight fairly."),
        ("get", "He gets results."),
        ("hang", "I hang art."),
        ("learn", "She learns quickly."),
        ("smell", "It smells good."),
        ("spin", "Wheels spin fast."),
        ("stick", "It sticks well."),
        ("swing", "Gates swing open."),
        ("work", "I work hard."),
        ("play", "She plays music."),
        ("study", "He studies daily."),
        ("travel", "They travel often."),
        ("visit", "I visit friends."),
        ("watch", "She watches closely."),
        ("listen", "He listens well."),
        ("talk", "They talk openly."),
        ("look", "I look forward."),
        ("seem", "She seems happy."),
        ("appear", "He appears confident."),
        ("happen", "Things happen naturally."),
        ("change", "Times change."),
        ("move", "I move forward."),
        ("live", "She lives fully."),
        ("die", "Plants die eventually."),
        ("smile", "They smile warmly."),
        ("laugh", "I laugh easily."),
        ("cry", "She cries rarely."),
        ("shout", "He shouts loudly."),
        ("whisper", "They whisper softly."),
        ("answer", "I answer honestly."),
        ("ask", "She asks questions."),
        ("call", "He calls regularly."),
        ("reply", "They reply quickly."),
        ("agree", "I agree strongly."),
        ("argue", "She argues logically."),
        ("believe", "He believes firmly."),
        ("decide", "They decide carefully."),
        ("explain", "I explain clearly."),
        ("help", "She helps others."),
        ("hope", "He hopes sincerely."),
        ("imagine", "They imagine freely."),
        ("jump", "I jump high."),
        ("kick", "She kicks hard."),
        ("kiss", "He kisses gently."),
        ("like", "They like it."),
        ("love", "I love deeply."),
        ("miss", "She misses them."),
        ("need", "He needs help."),
        ("notice", "They notice details."),
        ("offer", "I offer assistance."),
        ("open", "She opens doors."),
        ("close", "He closes windows."),
        ("pass", "They pass tests."),
        ("pull", "I pull gently."),
        ("push", "She pushes hard."),
        ("reach", "He reaches goals."),
        ("receive", "They receive gifts."),
        ("remember", "I remember clearly."),
        ("repeat", "She repeats often."),
        ("return", "He returns home."),
        ("save", "They save money."),
        ("search", "I search carefully."),
        ("stay", "She stays calm."),
        ("stop", "He stops quickly."),
        ("try", "They try hard."),
        ("turn", "I turn pages."),
        ("use", "She uses tools."),
        ("wait", "He waits patiently."),
        ("walks", "They walks daily."),
        ("want", "I want more."),
        ("wash", "She washes dishes."),
        ("wish", "He wishes well."),
        ("wonder", "They wonder often."),
        ("worry", "I worry sometimes."),
        ("yell", "She yells loudly."),
        ("enjoy", "He enjoys life."),
        ("finish", "They finish strong."),
        ("start", "I start fresh."),
        ("continue", "She continues bravely."),
        ("follow", "He follows closely."),
        ("guide", "They guide well."),
        ("join", "I join groups."),
        ("create", "She creates art."),
        ("destroy", "He destroys myths."),
        ("improve", "They improve daily."),
        ("increase", "Prices increase."),
        ("decrease", "Costs decrease."),
        ("develop", "She develops skills."),
        ("discover", "He discovers truths."),
        ("invent", "They invent things."),
        ("produce", "I produce results."),
        ("protect", "She protects others."),
        ("provide", "He provides support."),
        ("raise", "They raise standards."),
        ("reduce", "I reduce waste."),
        ("remove", "She removes obstacles."),
        ("replace", "He replaces parts."),
        ("require", "They require effort."),
        ("serve", "I serve customers."),
        ("share", "She shares freely."),
        ("suffer", "He suffers pain."),
        ("suggest", "They suggest ideas."),
        ("support", "I support causes."),
        ("suppose", "She supposes correctly."),
        ("surprise", "He surprises me."),
        ("survive", "They survive challenges."),
        ("touch", "I touch hearts."),
        ("treat", "She treats fairly."),
        ("trust", "He trusts completely."),
        ("value", "They value honesty."),
        ("vote", "I vote regularly."),
        ("warn", "She warns others."),
        ("welcome", "He welcomes guests."),
    ]

    for verb, text in present_verbs:
        dataset.append({
            'text': text,
            'target_word': verb,
            'label': 1
        })

    # Future tense verbs (200 examples)
    future_verbs = [
        ("walk", "I will walk tomorrow."),
        ("run", "She will run later."),
        ("eat", "They will eat soon."),
        ("write", "He will write next."),
        ("read", "I will read tonight."),
        ("speak", "She will speak tomorrow."),
        ("go", "They will go eventually."),
        ("come", "He will come later."),
        ("see", "I will see soon."),
        ("know", "She will know eventually."),
        ("think", "He will think about it."),
        ("feel", "They will feel better."),
        ("become", "She will become great."),
        ("leave", "I will leave tomorrow."),
        ("bring", "He will bring supplies."),
        ("begin", "The show will begin."),
        ("keep", "She will keep trying."),
        ("hold", "They will hold elections."),
        ("hear", "I will hear news."),
        ("let", "She will let you."),
        ("mean", "He will mean it."),
        ("meet", "They will meet soon."),
        ("pay", "I will pay later."),
        ("sit", "She will sit here."),
        ("stand", "He will stand firm."),
        ("understand", "They will understand."),
        ("win", "She will win eventually."),
        ("lose", "He will lose weight."),
        ("build", "They will build tomorrow."),
        ("buy", "I will buy later."),
        ("catch", "She will catch up."),
        ("choose", "He will choose wisely."),
        ("draw", "They will draw soon."),
        ("drive", "I will drive tomorrow."),
        ("fall", "She will fall asleep."),
        ("fly", "Birds will fly away."),
        ("forget", "He will forget eventually."),
        ("freeze", "It will freeze tonight."),
        ("give", "She will give generously."),
        ("grow", "They will grow quickly."),
        ("hide", "I will hide there."),
        ("hit", "She will hit targets."),
        ("hurt", "He will hurt less."),
        ("lay", "They will lay plans."),
        ("lead", "She will lead soon."),
        ("lend", "He will lend support."),
        ("light", "I will light candles."),
        ("make", "She will make dinner."),
        ("ride", "He will ride tomorrow."),
        ("sing", "They will sing tonight."),
        ("send", "I will send it."),
        ("shake", "She will shake hands."),
        ("shoot", "He will shoot photos."),
        ("show", "They will show up."),
        ("shut", "I will shut doors."),
        ("sleep", "She will sleep well."),
        ("spend", "He will spend time."),
        ("split", "They will split costs."),
        ("spread", "I will spread word."),
        ("steal", "Someone will steal."),
        ("strike", "Lightning will strike."),
        ("swim", "She will swim tomorrow."),
        ("swear", "He will swear in."),
        ("teach", "They will teach soon."),
        ("throw", "I will throw later."),
        ("tell", "She will tell stories."),
        ("take", "He will take notes."),
        ("tear", "They will tear down."),
        ("wake", "I will wake early."),
        ("wear", "She will wear blue."),
        ("find", "He will find it."),
        ("sell", "They will sell soon."),
        ("break", "I will break records."),
        ("drink", "She will drink water."),
        ("fight", "They will fight back."),
        ("get", "He will get better."),
        ("hang", "I will hang pictures."),
        ("learn", "She will learn quickly."),
        ("smell", "It will smell good."),
        ("spin", "Wheels will spin."),
        ("stick", "It will stick."),
        ("swing", "Gates will swing."),
        ("work", "I will work tomorrow."),
        ("play", "She will play music."),
        ("study", "He will study tonight."),
        ("travel", "They will travel soon."),
        ("visit", "I will visit later."),
        ("watch", "She will watch closely."),
        ("listen", "He will listen well."),
        ("talk", "They will talk soon."),
        ("look", "I will look forward."),
        ("seem", "She will seem happy."),
        ("appear", "He will appear soon."),
        ("happen", "Things will happen."),
        ("change", "Times will change."),
        ("move", "I will move forward."),
        ("live", "She will live fully."),
        ("die", "Plants will die."),
        ("smile", "They will smile."),
        ("laugh", "I will laugh."),
        ("cry", "She will cry less."),
        ("shout", "He will shout loudly."),
        ("whisper", "They will whisper."),
        ("answer", "I will answer honestly."),
        ("ask", "She will ask questions."),
        ("call", "He will call later."),
        ("reply", "They will reply soon."),
        ("agree", "I will agree."),
        ("argue", "She will argue logically."),
        ("believe", "He will believe."),
        ("decide", "They will decide."),
        ("explain", "I will explain."),
        ("help", "She will help."),
        ("hope", "He will hope."),
        ("imagine", "They will imagine."),
        ("jump", "I will jump."),
        ("kick", "She will kick."),
        ("kiss", "He will kiss."),
        ("like", "They will like it."),
        ("love", "I will love."),
        ("miss", "She will miss."),
        ("need", "He will need."),
        ("notice", "They will notice."),
        ("offer", "I will offer."),
        ("open", "She will open."),
        ("close", "He will close."),
        ("pass", "They will pass."),
        ("pull", "I will pull."),
        ("push", "She will push."),
        ("reach", "He will reach."),
        ("receive", "They will receive."),
        ("remember", "I will remember."),
        ("repeat", "She will repeat."),
        ("return", "He will return."),
        ("save", "They will save."),
        ("search", "I will search."),
        ("stay", "She will stay."),
        ("stop", "He will stop."),
        ("try", "They will try."),
        ("turn", "I will turn."),
        ("use", "She will use."),
        ("wait", "He will wait."),
        ("walk", "They will walk."),
        ("want", "I will want."),
        ("wash", "She will wash."),
        ("wish", "He will wish."),
        ("wonder", "They will wonder."),
        ("worry", "I will worry."),
        ("yell", "She will yell."),
        ("enjoy", "He will enjoy."),
        ("finish", "They will finish."),
        ("start", "I will start."),
        ("continue", "She will continue."),
        ("follow", "He will follow."),
        ("guide", "They will guide."),
        ("join", "I will join."),
        ("create", "She will create."),
        ("destroy", "He will destroy."),
        ("improve", "They will improve."),
        ("increase", "Prices will increase."),
        ("decrease", "Costs will decrease."),
        ("develop", "She will develop."),
        ("discover", "He will discover."),
        ("invent", "They will invent."),
        ("produce", "I will produce."),
        ("protect", "She will protect."),
        ("provide", "He will provide."),
        ("raise", "They will raise."),
        ("reduce", "I will reduce."),
        ("remove", "She will remove."),
        ("replace", "He will replace."),
        ("require", "They will require."),
        ("serve", "I will serve."),
        ("share", "She will share."),
        ("suffer", "He will suffer."),
        ("suggest", "They will suggest."),
        ("support", "I will support."),
        ("suppose", "She will suppose."),
        ("surprise", "He will surprise."),
        ("survive", "They will survive."),
        ("touch", "I will touch."),
        ("treat", "She will treat."),
        ("trust", "He will trust."),
        ("value", "They will value."),
        ("vote", "I will vote."),
        ("warn", "She will warn."),
        ("welcome", "He will welcome."),
    ]

    for verb, text in future_verbs:
        dataset.append({
            'text': text,
            'target_word': verb,
            'label': 2
        })

    return dataset


def create_sentiment_dataset() -> List[Dict]:
    """
    Create dataset for sentiment prediction task.

    Returns 200 unique examples for each sentiment (600 total).

    Returns:
        List of 600 examples with 'text', 'target_word', and 'label'
        (0=positive, 1=negative, 2=neutral)
    """
    dataset = []

    # Positive sentiment words (200 examples)
    positive_words = [
        ("amazing", "This is truly amazing."),
        ("wonderful", "What a wonderful day."),
        ("excellent", "The service was excellent."),
        ("fantastic", "That sounds fantastic."),
        ("great", "This is really great."),
        ("beautiful", "The view is beautiful."),
        ("perfect", "Everything went perfect."),
        ("brilliant", "That's a brilliant idea."),
        ("outstanding", "Their performance was outstanding."),
        ("superb", "The quality is superb."),
        ("delightful", "How delightful this is."),
        ("magnificent", "A truly magnificent sight."),
        ("marvelous", "The results are marvelous."),
        ("splendid", "What a splendid evening."),
        ("terrific", "That's terrific news."),
        ("fabulous", "She looks fabulous today."),
        ("gorgeous", "The sunset is gorgeous."),
        ("lovely", "What a lovely gesture."),
        ("charming", "He is quite charming."),
        ("delicious", "The food tastes delicious."),
        ("enjoyable", "This is very enjoyable."),
        ("pleasant", "A pleasant surprise indeed."),
        ("satisfying", "Very satisfying results."),
        ("impressive", "That's truly impressive."),
        ("remarkable", "A remarkable achievement."),
        ("exceptional", "Of exceptional quality."),
        ("incredible", "Simply incredible work."),
        ("awesome", "This is totally awesome."),
        ("stunning", "Absolutely stunning views."),
        ("spectacular", "A spectacular performance."),
        ("admirable", "Truly admirable effort."),
        ("phenomenal", "The growth is phenomenal."),
        ("refreshing", "How refreshing this feels."),
        ("exciting", "This is so exciting."),
        ("thrilling", "What a thrilling experience."),
        ("inspiring", "Very inspiring message."),
        ("uplifting", "Such uplifting words."),
        ("cheerful", "She seems very cheerful."),
        ("joyful", "A joyful celebration."),
        ("happy", "Everyone looks happy today."),
        ("glad", "I'm so glad to hear."),
        ("pleased", "Very pleased with results."),
        ("content", "Feeling quite content now."),
        ("grateful", "We are deeply grateful."),
        ("thankful", "So thankful for this."),
        ("blessed", "We feel truly blessed."),
        ("fortunate", "How fortunate we are."),
        ("lucky", "We got very lucky."),
        ("optimistic", "Feeling quite optimistic."),
        ("hopeful", "Remaining hopeful still."),
        ("confident", "Very confident about this."),
        ("positive", "Maintaining a positive outlook."),
        ("enthusiastic", "Quite enthusiastic about it."),
        ("passionate", "She's very passionate."),
        ("eager", "So eager to begin."),
        ("keen", "Very keen to participate."),
        ("interested", "Deeply interested in this."),
        ("curious", "Quite curious about it."),
        ("fascinated", "Completely fascinated by this."),
        ("intrigued", "Very intrigued indeed."),
        ("captivated", "Totally captivated by it."),
        ("enchanted", "Simply enchanted by this."),
        ("mesmerized", "Absolutely mesmerized."),
        ("energetic", "Feeling very energetic today."),
        ("vibrant", "Such a vibrant atmosphere."),
        ("lively", "The party was lively."),
        ("dynamic", "A dynamic presentation."),
        ("spirited", "Very spirited performance."),
        ("animated", "An animated discussion."),
        ("fun", "This is really fun."),
        ("entertaining", "Very entertaining show."),
        ("amusing", "Quite amusing story."),
        ("hilarious", "That was hilarious."),
        ("funny", "Something very funny."),
        ("witty", "A witty remark."),
        ("clever", "That's quite clever."),
        ("smart", "Very smart thinking."),
        ("wise", "A wise decision."),
        ("intelligent", "Highly intelligent approach."),
        ("brilliant", "A brilliant strategy."),
        ("genius", "That's pure genius."),
        ("talented", "She's incredibly talented."),
        ("skilled", "Very skilled craftsman."),
        ("capable", "Highly capable team."),
        ("competent", "A competent leader."),
        ("efficient", "Very efficient system."),
        ("effective", "Highly effective method."),
        ("productive", "A productive meeting."),
        ("successful", "The project was successful."),
        ("victorious", "The team was victorious."),
        ("triumphant", "A triumphant return."),
        ("winning", "The winning strategy."),
        ("champion", "A true champion."),
        ("accomplished", "Highly accomplished artist."),
        ("distinguished", "A distinguished guest."),
        ("prestigious", "A prestigious award."),
        ("renowned", "The renowned expert."),
        ("famous", "A famous landmark."),
        ("popular", "Very popular choice."),
        ("beloved", "A beloved character."),
        ("cherished", "A cherished memory."),
        ("treasured", "A treasured possession."),
        ("valued", "A valued member."),
        ("precious", "Such a precious moment."),
        ("special", "A very special day."),
        ("unique", "Truly unique experience."),
        ("rare", "A rare opportunity."),
        ("extraordinary", "An extraordinary talent."),
        ("uncommon", "An uncommon sight."),
        ("unusual", "An unusual but good thing."),
        ("original", "Very original idea."),
        ("creative", "A creative solution."),
        ("innovative", "An innovative approach."),
        ("fresh", "A fresh perspective."),
        ("new", "Something new and good."),
        ("novel", "A novel concept."),
        ("modern", "Very modern design."),
        ("contemporary", "A contemporary masterpiece."),
        ("stylish", "Quite stylish outfit."),
        ("elegant", "An elegant solution."),
        ("graceful", "Such graceful movements."),
        ("refined", "A refined taste."),
        ("sophisticated", "Very sophisticated approach."),
        ("polished", "A polished performance."),
        ("professional", "Very professional work."),
        ("reliable", "A reliable partner."),
        ("dependable", "Highly dependable service."),
        ("trustworthy", "A trustworthy source."),
        ("honest", "An honest assessment."),
        ("sincere", "A sincere apology."),
        ("genuine", "A genuine smile."),
        ("authentic", "An authentic experience."),
        ("real", "The real deal."),
        ("true", "A true friend."),
        ("loyal", "A loyal companion."),
        ("faithful", "A faithful servant."),
        ("devoted", "A devoted fan."),
        ("dedicated", "Very dedicated worker."),
        ("committed", "Highly committed team."),
        ("determined", "A determined effort."),
        ("persistent", "Very persistent approach."),
        ("resilient", "A resilient spirit."),
        ("strong", "A strong character."),
        ("powerful", "A powerful message."),
        ("mighty", "A mighty force."),
        ("robust", "A robust system."),
        ("sturdy", "A sturdy construction."),
        ("solid", "A solid foundation."),
        ("stable", "A stable platform."),
        ("secure", "A secure environment."),
        ("safe", "A safe place."),
        ("protected", "Feeling protected here."),
        ("comfortable", "Very comfortable setting."),
        ("cozy", "Such a cozy atmosphere."),
        ("warm", "A warm welcome."),
        ("friendly", "Very friendly staff."),
        ("kind", "A kind gesture."),
        ("gentle", "A gentle touch."),
        ("caring", "A caring approach."),
        ("compassionate", "A compassionate response."),
        ("thoughtful", "A thoughtful gift."),
        ("considerate", "Very considerate behavior."),
        ("generous", "A generous offer."),
        ("giving", "A giving nature."),
        ("helpful", "Very helpful advice."),
        ("supportive", "A supportive environment."),
        ("encouraging", "Encouraging words."),
        ("motivating", "A motivating speech."),
        ("empowering", "An empowering message."),
        ("liberating", "A liberating experience."),
        ("freeing", "A freeing thought."),
        ("peaceful", "A peaceful scene."),
        ("calm", "A calm demeanor."),
        ("serene", "A serene landscape."),
        ("tranquil", "A tranquil setting."),
        ("relaxed", "Feeling very relaxed."),
        ("soothing", "A soothing voice."),
        ("comforting", "A comforting presence."),
        ("reassuring", "A reassuring message."),
        ("promising", "A promising start."),
        ("bright", "A bright future."),
        ("radiant", "A radiant smile."),
        ("glowing", "A glowing review."),
        ("shining", "A shining example."),
        ("sparkling", "A sparkling performance."),
        ("dazzling", "A dazzling display."),
        ("gleaming", "A gleaming surface."),
        ("lustrous", "A lustrous finish."),
        ("pristine", "In pristine condition."),
        ("immaculate", "An immaculate presentation."),
        ("flawless", "A flawless execution."),
        ("impeccable", "Impeccable timing."),
    ]

    for word, text in positive_words:
        dataset.append({
            'text': text,
            'target_word': word,
            'label': 0
        })

    # Negative sentiment words (200 examples)
    negative_words = [
        ("terrible", "This is absolutely terrible."),
        ("awful", "The weather is awful."),
        ("horrible", "What a horrible experience."),
        ("dreadful", "The service was dreadful."),
        ("bad", "This is really bad."),
        ("poor", "The quality is poor."),
        ("disappointing", "Very disappointing results."),
        ("unpleasant", "An unpleasant surprise."),
        ("disgusting", "That smells disgusting."),
        ("nasty", "A nasty comment."),
        ("ugly", "The design is ugly."),
        ("hideous", "A hideous sight."),
        ("revolting", "Absolutely revolting taste."),
        ("repulsive", "A repulsive behavior."),
        ("offensive", "That's quite offensive."),
        ("annoying", "This is so annoying."),
        ("irritating", "Very irritating noise."),
        ("frustrating", "How frustrating this is."),
        ("aggravating", "An aggravating situation."),
        ("infuriating", "Simply infuriating behavior."),
        ("maddening", "A maddening problem."),
        ("unbearable", "The pain is unbearable."),
        ("intolerable", "An intolerable situation."),
        ("unacceptable", "This is unacceptable."),
        ("inadequate", "The response was inadequate."),
        ("insufficient", "Insufficient evidence provided."),
        ("deficient", "A deficient system."),
        ("lacking", "Clearly lacking quality."),
        ("inferior", "Of inferior quality."),
        ("substandard", "Substandard performance."),
        ("mediocre", "A mediocre attempt."),
        ("pathetic", "That's just pathetic."),
        ("pitiful", "A pitiful display."),
        ("miserable", "Feeling absolutely miserable."),
        ("depressing", "A depressing situation."),
        ("gloomy", "Such a gloomy atmosphere."),
        ("bleak", "The outlook is bleak."),
        ("dismal", "A dismal failure."),
        ("dreary", "A dreary day."),
        ("sad", "This makes me sad."),
        ("unhappy", "Everyone seems unhappy."),
        ("sorrowful", "A sorrowful occasion."),
        ("mournful", "A mournful song."),
        ("tragic", "A tragic event."),
        ("heartbreaking", "Truly heartbreaking news."),
        ("devastating", "A devastating loss."),
        ("crushing", "A crushing defeat."),
        ("painful", "A painful experience."),
        ("agonizing", "An agonizing decision."),
        ("excruciating", "Excruciating pain."),
        ("torturous", "A torturous process."),
        ("harsh", "Very harsh criticism."),
        ("severe", "A severe problem."),
        ("cruel", "A cruel punishment."),
        ("brutal", "A brutal attack."),
        ("savage", "A savage response."),
        ("vicious", "A vicious cycle."),
        ("mean", "That was mean."),
        ("unkind", "An unkind remark."),
        ("rude", "Very rude behavior."),
        ("impolite", "An impolite response."),
        ("disrespectful", "Highly disrespectful attitude."),
        ("insulting", "That's insulting."),
        ("degrading", "A degrading experience."),
        ("humiliating", "A humiliating moment."),
        ("embarrassing", "How embarrassing this is."),
        ("shameful", "A shameful act."),
        ("disgraceful", "Disgraceful behavior."),
        ("scandalous", "A scandalous revelation."),
        ("outrageous", "That's outrageous."),
        ("shocking", "Shocking news indeed."),
        ("appalling", "Appalling conditions."),
        ("alarming", "An alarming trend."),
        ("disturbing", "A disturbing pattern."),
        ("troubling", "Very troubling signs."),
        ("worrying", "A worrying development."),
        ("concerning", "This is concerning."),
        ("problematic", "A problematic situation."),
        ("difficult", "A difficult challenge."),
        ("hard", "This is too hard."),
        ("tough", "A tough day."),
        ("challenging", "Very challenging task."),
        ("demanding", "A demanding schedule."),
        ("exhausting", "An exhausting process."),
        ("tiring", "Very tiring work."),
        ("draining", "An emotionally draining experience."),
        ("taxing", "A taxing ordeal."),
        ("stressful", "A stressful situation."),
        ("overwhelming", "This feels overwhelming."),
        ("burdensome", "A burdensome responsibility."),
        ("oppressive", "An oppressive atmosphere."),
        ("suffocating", "A suffocating environment."),
        ("confining", "A confining space."),
        ("restrictive", "Very restrictive rules."),
        ("limiting", "A limiting factor."),
        ("constraining", "Constraining circumstances."),
        ("inhibiting", "An inhibiting presence."),
        ("hindering", "A hindering factor."),
        ("obstructive", "An obstructive behavior."),
        ("interfering", "Stop interfering please."),
        ("disruptive", "A disruptive influence."),
        ("chaotic", "The situation is chaotic."),
        ("messy", "A messy situation."),
        ("disorganized", "Completely disorganized effort."),
        ("confused", "I'm very confused."),
        ("perplexed", "Quite perplexed by this."),
        ("bewildered", "Feeling bewildered now."),
        ("puzzled", "Very puzzled indeed."),
        ("uncertain", "An uncertain future."),
        ("doubtful", "Feeling quite doubtful."),
        ("skeptical", "Very skeptical about this."),
        ("suspicious", "A suspicious activity."),
        ("distrustful", "Feeling distrustful."),
        ("wary", "Very wary of this."),
        ("cautious", "Overly cautious approach."),
        ("hesitant", "Feeling quite hesitant."),
        ("reluctant", "Very reluctant to proceed."),
        ("unwilling", "Completely unwilling to help."),
        ("resistant", "Resistant to change."),
        ("opposed", "Strongly opposed to this."),
        ("against", "I'm against this."),
        ("contrary", "A contrary opinion."),
        ("contradictory", "Contradictory statements."),
        ("inconsistent", "Very inconsistent results."),
        ("unreliable", "An unreliable source."),
        ("undependable", "Completely undependable."),
        ("untrustworthy", "An untrustworthy person."),
        ("dishonest", "Dishonest practices."),
        ("deceitful", "A deceitful scheme."),
        ("misleading", "Misleading information."),
        ("false", "That's completely false."),
        ("fake", "A fake product."),
        ("fraudulent", "Fraudulent claims."),
        ("corrupt", "A corrupt system."),
        ("unethical", "Unethical behavior."),
        ("immoral", "An immoral act."),
        ("wrong", "This is just wrong."),
        ("evil", "An evil plan."),
        ("wicked", "A wicked deed."),
        ("sinful", "A sinful thought."),
        ("guilty", "Feeling very guilty."),
        ("ashamed", "Deeply ashamed."),
        ("regretful", "Very regretful now."),
        ("remorseful", "Feeling remorseful."),
        ("sorry", "So sorry about this."),
        ("apologetic", "An apologetic tone."),
        ("weak", "A weak argument."),
        ("feeble", "A feeble attempt."),
        ("frail", "A frail structure."),
        ("fragile", "Very fragile condition."),
        ("delicate", "A delicate but bad situation."),
        ("vulnerable", "Feeling very vulnerable."),
        ("exposed", "Feeling quite exposed."),
        ("defenseless", "Completely defenseless."),
        ("helpless", "Feeling utterly helpless."),
        ("powerless", "Absolutely powerless."),
        ("impotent", "An impotent response."),
        ("ineffective", "An ineffective solution."),
        ("useless", "This is useless."),
        ("worthless", "A worthless effort."),
        ("pointless", "This seems pointless."),
        ("futile", "A futile attempt."),
        ("hopeless", "The situation is hopeless."),
        ("desperate", "A desperate situation."),
        ("dire", "In dire circumstances."),
        ("critical", "A critical failure."),
        ("dangerous", "A dangerous path."),
        ("hazardous", "Hazardous conditions."),
        ("risky", "A risky decision."),
        ("perilous", "A perilous journey."),
        ("threatening", "A threatening message."),
        ("menacing", "A menacing presence."),
        ("ominous", "An ominous sign."),
        ("foreboding", "A foreboding feeling."),
        ("sinister", "A sinister plot."),
        ("dark", "A dark chapter."),
        ("grim", "The outlook is grim."),
        ("sombre", "A sombre mood."),
        ("melancholy", "A melancholy song."),
        ("lonely", "Feeling very lonely."),
        ("isolated", "Completely isolated."),
        ("abandoned", "Feeling abandoned."),
        ("neglected", "A neglected issue."),
        ("forgotten", "A forgotten promise."),
        ("ignored", "Feeling ignored."),
        ("rejected", "Feeling rejected."),
        ("excluded", "Feeling excluded."),
        ("unwanted", "Feeling unwanted."),
        ("unloved", "Feeling unloved."),
        ("hated", "Feeling hated."),
        ("despised", "Feeling despised."),
        ("loathed", "Absolutely loathed."),
        ("detested", "Utterly detested."),
    ]

    for word, text in negative_words:
        dataset.append({
            'text': text,
            'target_word': word,
            'label': 1
        })

    # Neutral sentiment words (200 examples)
    neutral_words = [
        ("normal", "This seems quite normal."),
        ("average", "An average performance."),
        ("ordinary", "Just an ordinary day."),
        ("regular", "A regular occurrence."),
        ("standard", "The standard procedure."),
        ("typical", "A typical response."),
        ("usual", "The usual routine."),
        ("common", "A common practice."),
        ("general", "In general terms."),
        ("basic", "The basic requirements."),
        ("simple", "A simple explanation."),
        ("plain", "A plain statement."),
        ("moderate", "A moderate approach."),
        ("medium", "A medium size."),
        ("neutral", "Maintaining a neutral stance."),
        ("balanced", "A balanced view."),
        ("even", "An even distribution."),
        ("fair", "A fair assessment."),
        ("equal", "Equal opportunities."),
        ("consistent", "A consistent pattern."),
        ("steady", "A steady pace."),
        ("stable", "A stable condition."),
        ("constant", "A constant rate."),
        ("uniform", "A uniform appearance."),
        ("similar", "Similar results."),
        ("alike", "They look alike."),
        ("comparable", "Comparable outcomes."),
        ("equivalent", "An equivalent value."),
        ("same", "The same as before."),
        ("identical", "Identical copies."),
        ("unchanged", "Remains unchanged."),
        ("static", "A static situation."),
        ("fixed", "A fixed amount."),
        ("set", "A set schedule."),
        ("established", "An established fact."),
        ("known", "A known quantity."),
        ("familiar", "A familiar sight."),
        ("recognized", "A recognized pattern."),
        ("expected", "An expected outcome."),
        ("predicted", "As predicted earlier."),
        ("anticipated", "An anticipated result."),
        ("planned", "According to plan."),
        ("scheduled", "The scheduled time."),
        ("arranged", "As arranged previously."),
        ("organized", "An organized system."),
        ("systematic", "A systematic approach."),
        ("methodical", "A methodical process."),
        ("orderly", "An orderly arrangement."),
        ("structured", "A structured format."),
        ("formal", "A formal procedure."),
        ("official", "The official statement."),
        ("conventional", "A conventional method."),
        ("traditional", "A traditional approach."),
        ("customary", "The customary practice."),
        ("routine", "A routine check."),
        ("habitual", "A habitual action."),
        ("mechanical", "A mechanical process."),
        ("automatic", "An automatic response."),
        ("programmed", "A programmed sequence."),
        ("calculated", "A calculated move."),
        ("measured", "A measured response."),
        ("precise", "A precise measurement."),
        ("exact", "The exact amount."),
        ("accurate", "An accurate description."),
        ("correct", "The correct answer."),
        ("proper", "The proper procedure."),
        ("appropriate", "An appropriate response."),
        ("suitable", "A suitable option."),
        ("adequate", "An adequate supply."),
        ("sufficient", "Sufficient evidence."),
        ("enough", "That's enough now."),
        ("satisfactory", "A satisfactory result."),
        ("acceptable", "An acceptable solution."),
        ("passable", "A passable effort."),
        ("tolerable", "A tolerable situation."),
        ("bearable", "A bearable condition."),
        ("manageable", "A manageable task."),
        ("feasible", "A feasible plan."),
        ("possible", "It's quite possible."),
        ("potential", "A potential option."),
        ("probable", "A probable outcome."),
        ("likely", "That's quite likely."),
        ("plausible", "A plausible explanation."),
        ("reasonable", "A reasonable request."),
        ("rational", "A rational approach."),
        ("logical", "A logical conclusion."),
        ("sensible", "A sensible decision."),
        ("practical", "A practical solution."),
        ("realistic", "A realistic goal."),
        ("achievable", "An achievable target."),
        ("attainable", "An attainable objective."),
        ("reachable", "A reachable destination."),
        ("accessible", "An accessible location."),
        ("available", "Currently available."),
        ("present", "Those present today."),
        ("existing", "The existing system."),
        ("current", "The current situation."),
        ("ongoing", "An ongoing process."),
        ("continuing", "A continuing effort."),
        ("persistent", "A persistent issue."),
        ("remaining", "The remaining items."),
        ("leftover", "Some leftover materials."),
        ("residual", "A residual effect."),
        ("partial", "A partial solution."),
        ("incomplete", "An incomplete picture."),
        ("limited", "A limited supply."),
        ("restricted", "Restricted access."),
        ("confined", "Confined to this area."),
        ("contained", "Contained within bounds."),
        ("enclosed", "An enclosed space."),
        ("surrounded", "Surrounded by walls."),
        ("bordered", "Bordered by trees."),
        ("adjacent", "An adjacent room."),
        ("nearby", "A nearby location."),
        ("close", "Close to here."),
        ("proximate", "In proximate area."),
        ("neighboring", "A neighboring town."),
        ("local", "A local business."),
        ("regional", "A regional office."),
        ("national", "A national standard."),
        ("international", "An international agreement."),
        ("global", "A global perspective."),
        ("universal", "A universal truth."),
        ("widespread", "A widespread practice."),
        ("extensive", "An extensive network."),
        ("broad", "A broad category."),
        ("wide", "A wide range."),
        ("large", "A large amount."),
        ("big", "A big building."),
        ("substantial", "A substantial portion."),
        ("considerable", "A considerable number."),
        ("significant", "A significant amount."),
        ("notable", "A notable feature."),
        ("marked", "A marked difference."),
        ("distinct", "A distinct characteristic."),
        ("separate", "A separate issue."),
        ("different", "A different approach."),
        ("various", "Various options available."),
        ("diverse", "A diverse group."),
        ("multiple", "Multiple factors involved."),
        ("several", "Several attempts made."),
        ("numerous", "Numerous examples exist."),
        ("many", "Many people attended."),
        ("few", "A few remain."),
        ("some", "Some prefer this."),
        ("certain", "Certain conditions apply."),
        ("specific", "A specific requirement."),
        ("particular", "A particular case."),
        ("individual", "An individual choice."),
        ("single", "A single instance."),
        ("sole", "The sole purpose."),
        ("only", "The only option."),
        ("exclusive", "An exclusive feature."),
        ("unique", "Each unique in nature."),
        ("special", "No special treatment."),
        ("distinctive", "A distinctive characteristic."),
        ("characteristic", "A characteristic feature."),
        ("typical", "A typical example."),
        ("representative", "A representative sample."),
        ("indicative", "Indicative of pattern."),
        ("suggestive", "Suggestive of trend."),
        ("symbolic", "A symbolic gesture."),
        ("figurative", "A figurative expression."),
        ("literal", "A literal interpretation."),
        ("actual", "The actual fact."),
        ("real", "The real situation."),
        ("true", "A true statement."),
        ("genuine", "A genuine article."),
        ("authentic", "An authentic document."),
        ("legitimate", "A legitimate concern."),
        ("valid", "A valid point."),
        ("sound", "A sound argument."),
        ("solid", "A solid basis."),
        ("firm", "A firm position."),
        ("definite", "A definite answer."),
        ("clear", "A clear statement."),
        ("obvious", "An obvious fact."),
        ("apparent", "An apparent trend."),
        ("evident", "An evident pattern."),
        ("visible", "A visible marker."),
        ("noticeable", "A noticeable change."),
        ("observable", "An observable phenomenon."),
        ("detectable", "A detectable signal."),
        ("measurable", "A measurable effect."),
        ("quantifiable", "A quantifiable result."),
        ("objective", "An objective measure."),
        ("factual", "A factual report."),
        ("empirical", "Empirical evidence."),
        ("concrete", "A concrete example."),
        ("tangible", "A tangible benefit."),
        ("material", "Material evidence."),
        ("physical", "Physical characteristics."),
        ("natural", "A natural occurrence."),
        ("organic", "An organic process."),
    ]

    for word, text in neutral_words:
        dataset.append({
            'text': text,
            'target_word': word,
            'label': 2
        })

    return dataset


def find_target_token_position(
    tokens: torch.Tensor,
    tokenizer,
    text: str,
    target_word: str
) -> int:
    """
    Find the last token position of the target word in the tokenized text.

    Args:
        tokens: Tokenized input (1, seq_len)
        tokenizer: Model tokenizer
        text: Original text
        target_word: Target word to find

    Returns:
        Position index of the last token of the target word
    """
    # Decode all tokens
    token_strs = [tokenizer.decode([t]) for t in tokens[0]]

    # Find target word in original text
    target_start = text.lower().find(target_word.lower())
    if target_start == -1:
        raise ValueError(f"Target word '{target_word}' not found in text '{text}'")

    # Reconstruct text from tokens to find position
    current_pos = 0
    reconstructed = ""

    for i, token_str in enumerate(token_strs):
        # Update reconstructed text
        reconstructed += token_str
        current_pos = len(reconstructed)

        # Check if we've passed the end of the target word
        if current_pos >= target_start + len(target_word):
            return i

    # If we didn't find it, return the last token
    return len(token_strs) - 1


def extract_activations(
    model,
    examples: List[Dict],
    layer: int,
    logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract activations at target token positions for all examples.

    Args:
        model: HookedTransformer model
        examples: List of examples with 'text', 'target_word', 'label'
        layer: Layer index to extract from
        logger: Logger instance

    Returns:
        Tuple of (activations, labels) as numpy arrays
        activations: (n_examples, 768)
        labels: (n_examples,)
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    activations_list = []
    labels_list = []

    for example in tqdm(examples, desc=f"Layer {layer} - Extracting"):
        text = example['text']
        target_word = example['target_word']
        label = example['label']

        # Tokenize
        tokens = model.to_tokens(text)

        # Find target position
        try:
            target_pos = find_target_token_position(
                tokens, model.tokenizer, text, target_word
            )
        except ValueError as e:
            logger.warning(f"Skipping example: {e}")
            continue

        # Run model and extract activations
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=[hook_name]
            )

            # Extract activation at target position
            # cache[hook_name] is (1, seq_len, d_model)
            activation = cache[hook_name][0, target_pos, :].cpu().numpy()

            activations_list.append(activation)
            labels_list.append(label)

    activations = np.array(activations_list)
    labels = np.array(labels_list)

    logger.info(f"  Extracted {len(activations)} activations of shape {activations.shape}")

    return activations, labels


def log_diagnostics(
    activations: np.ndarray,
    labels: np.ndarray,
    task_name: str,
    logger: logging.Logger = None
):
    """
    Log diagnostic information about activations and labels.

    Args:
        activations: (n_examples, d_model) activation matrix
        labels: (n_examples,) label array
        task_name: Name of the task for logging
        logger: Logger instance
    """
    if not logger:
        return

    logger.info(f"  [DIAGNOSTICS] {task_name}")
    logger.info(f"    Activation shape: {activations.shape}")
    logger.info(f"    Label shape: {labels.shape}")

    # Check activation statistics
    logger.info(f"    Activation mean: {activations.mean():.6f}")
    logger.info(f"    Activation std: {activations.std():.6f}")
    logger.info(f"    Activation min: {activations.min():.6f}")
    logger.info(f"    Activation max: {activations.max():.6f}")

    # Check for constant/near-constant activations
    activation_variance = activations.var(axis=0)
    zero_variance_dims = (activation_variance < 1e-10).sum()
    logger.info(f"    Dimensions with zero variance: {zero_variance_dims}/{activations.shape[1]}")

    # Check label distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    logger.info(f"    Label distribution: {dict(zip(unique_labels, counts))}")

    # Check if activations differ between classes
    if len(unique_labels) == 2:  # Binary classification
        class0_acts = activations[labels == unique_labels[0]]
        class1_acts = activations[labels == unique_labels[1]]

        mean_diff = np.abs(class0_acts.mean(axis=0) - class1_acts.mean(axis=0)).mean()
        logger.info(f"    Mean activation difference between classes: {mean_diff:.6f}")

        # Check if classes are separable at all
        total_var = activations.var()
        between_class_var = ((class0_acts.mean() - class1_acts.mean()) ** 2) / 2
        logger.info(f"    Total variance: {total_var:.6f}")
        logger.info(f"    Between-class variance: {between_class_var:.6f}")
        logger.info(f"    Separability ratio: {between_class_var / (total_var + 1e-10):.6f}")


def apply_pca_and_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    n_components: int = 10,
    n_runs: int = 3,
    logger: logging.Logger = None
) -> Dict:
    """
    Apply PCA and train probes to measure classification performance.

    Args:
        activations: (n_examples, 768) activation matrix
        labels: (n_examples,) label array
        n_components: Number of PCA components (default: 10)
        n_runs: Number of probe training runs (default: 3)
        logger: Logger instance

    Returns:
        Dictionary with:
        - explained_variance_ratio: per-component explained variance
        - cumulative_variance: cumulative explained variance
        - mutual_information: list of MI scores (one per run)
        - accuracy: list of accuracy scores (one per run)
        - f1_score: list of F1 scores (one per run)
    """
    # Standardize activations (mean=0, std=1 per feature)
    scaler = StandardScaler()
    standardized_activations = scaler.fit_transform(activations)

    # Fit PCA on standardized activations
    pca = PCA(n_components=n_components)
    reduced_activations = pca.fit_transform(standardized_activations)

    # Log explained variance
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    if logger:
        logger.info(f"  PCA explained variance (top {n_components} components):")
        logger.info(f"    Per-component: {explained_var}")
        logger.info(f"    Cumulative: {cumulative_var[-1]:.4f} ({cumulative_var[-1]*100:.1f}%)")

    # Train probes multiple times
    mi_scores = []
    accuracy_scores = []
    f1_scores = []

    for run in range(n_runs):
        # Train logistic regression probe with increased iterations
        probe = LogisticRegression(max_iter=2000, random_state=42 + run)
        probe.fit(reduced_activations, labels)

        # Get predictions
        predictions = probe.predict(reduced_activations)

        # Calculate metrics
        mi = mutual_info_score(labels, predictions)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='macro')

        mi_scores.append(mi)
        accuracy_scores.append(acc)
        f1_scores.append(f1)

    if logger:
        logger.info(f"  PCA Probe performance ({n_runs} runs):")
        logger.info(f"    Mutual Information: {np.mean(mi_scores):.4f} ± {np.std(mi_scores):.4f}")
        logger.info(f"    Accuracy: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
        logger.info(f"    F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

    return {
        'explained_variance_ratio': explained_var,
        'cumulative_variance': cumulative_var[-1],
        'mutual_information': mi_scores,
        'accuracy': accuracy_scores,
        'f1_score': f1_scores
    }


def apply_random_and_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    n_features: int = None,  # Ignored when using Gaussian sampling
    n_subsets: int = 20,
    logger: logging.Logger = None
) -> Dict:
    """
    Sample random feature subsets and train probes (baseline comparison).

    For each subset, sample the number of features from a Gaussian distribution
    centered at width/2 (384 for d_model=768), then randomly select that many features.
    Train one probe per random subset (20 subsets total for comparison).

    Args:
        activations: (n_examples, 768) activation matrix
        labels: (n_examples,) label array
        n_features: Ignored (kept for backward compatibility)
        n_subsets: Number of random subsets to try (default: 20)
        logger: Logger instance

    Returns:
        Dictionary with:
        - mutual_information: list of MI scores (one per subset)
        - accuracy: list of accuracy scores (one per subset)
        - f1_score: list of F1 scores (one per subset)
        - n_features_used: list of number of features used per subset
    """
    # Standardize activations first (mean=0, std=1 per feature)
    scaler = StandardScaler()
    standardized_activations = scaler.fit_transform(activations)

    d_model = standardized_activations.shape[1]  # Should be 768
    mean_features = d_model // 2  # 384 for d_model=768
    std_features = 50  # Standard deviation for Gaussian sampling

    mi_scores = []
    accuracy_scores = []
    f1_scores = []
    n_features_list = []

    for subset_idx in range(n_subsets):
        # Sample number of features from Gaussian distribution
        np.random.seed(42 + subset_idx)  # Reproducible
        n_features_sample = int(np.random.normal(mean_features, std_features))

        # Clip to valid range [10, d_model]
        n_features_sample = max(10, min(d_model, n_features_sample))
        n_features_list.append(n_features_sample)

        # Randomly sample features from standardized activations
        selected_features = np.random.choice(d_model, size=n_features_sample, replace=False)
        random_activations = standardized_activations[:, selected_features]

        # Train logistic regression probe with more iterations
        probe = LogisticRegression(max_iter=2000, random_state=42 + subset_idx)
        probe.fit(random_activations, labels)

        # Get predictions
        predictions = probe.predict(random_activations)

        # Calculate metrics
        mi = mutual_info_score(labels, predictions)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='macro')

        mi_scores.append(mi)
        accuracy_scores.append(acc)
        f1_scores.append(f1)

    if logger:
        logger.info(f"  Random baseline ({n_subsets} subsets, Gaussian features ~ N({mean_features}, {std_features})):")
        logger.info(f"    Feature counts: min={min(n_features_list)}, max={max(n_features_list)}, mean={np.mean(n_features_list):.1f}")
        logger.info(f"    Mutual Information: {np.mean(mi_scores):.4f} ± {np.std(mi_scores):.4f}")
        logger.info(f"    Accuracy: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
        logger.info(f"    F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

    return {
        'mutual_information': mi_scores,
        'accuracy': accuracy_scores,
        'f1_score': f1_scores,
        'n_features_used': n_features_list
    }


def create_bar_plot(
    results_df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
    logger: logging.Logger
):
    """
    Create bar plot with confidence intervals across layers.

    Args:
        results_df: DataFrame with results for all layers
        metric_col: Column name for the metric to plot
        ylabel: Y-axis label
        title: Plot title
        output_path: Path to save the plot
        logger: Logger instance
    """
    # Get unique layers from data
    layer_list = sorted(results_df['layer'].unique())
    means = []
    cis = []

    for layer in layer_list:
        layer_df = results_df[results_df['layer'] == layer]
        values = layer_df[metric_col].values

        # Calculate mean and 95% CI
        mean = values.mean()
        ci = stats.t.interval(
            confidence=0.95,
            df=len(values)-1,
            loc=mean,
            scale=stats.sem(values)
        )
        ci_error = mean - ci[0]  # Error bar size (symmetric)

        means.append(mean)
        cis.append(ci_error)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(layer_list, means, yerr=cis, capsize=5, alpha=0.7,
                   color='steelblue', ecolor='black', linewidth=1.5)

    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks(layer_list)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"  Saved: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Linear probe PCA experiment on GPT-2 Small"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/linear_probe_pca",
        help="Output directory for results"
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=10,
        help="Number of PCA components (default: 10)"
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=3,
        help="Number of probe training runs (default: 3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    logger.info("="*80)
    logger.info("LINEAR PROBE PCA EXPERIMENT")
    logger.info("="*80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"PCA components: {args.n_components}")
    logger.info(f"Probe runs: {args.n_runs}")
    logger.info(f"Random seed: {args.seed}")

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    logger.info("\n" + "="*80)
    logger.info("LOADING MODEL")
    logger.info("="*80)

    model_loader = ModelLoader(
        model_name="gpt2-small",
        layers=list(range(1, 12)),  # Layers 1-11 (skip layer 0 - input embeddings)
        device=device,
        logger=logger
    )

    model = model_loader.load_model()

    # Create datasets
    logger.info("\n" + "="*80)
    logger.info("CREATING DATASETS")
    logger.info("="*80)

    # SKIPPING plurality dataset - no signal (separability ratio = 0.0)
    # plurality_data = create_plurality_dataset()
    pos_data = create_pos_dataset()
    verb_tense_data = create_verb_tense_dataset()
    sentiment_data = create_sentiment_dataset()

    logger.info(f"POS dataset: {len(pos_data)} examples")
    logger.info(f"  Nouns: {sum(1 for x in pos_data if x['label'] == 0)}")
    logger.info(f"  Verbs: {sum(1 for x in pos_data if x['label'] == 1)}")
    logger.info(f"  Adjectives: {sum(1 for x in pos_data if x['label'] == 2)}")
    logger.info(f"  Adverbs: {sum(1 for x in pos_data if x['label'] == 3)}")

    logger.info(f"Verb Tense dataset: {len(verb_tense_data)} examples")
    logger.info(f"  Past: {sum(1 for x in verb_tense_data if x['label'] == 0)}")
    logger.info(f"  Present: {sum(1 for x in verb_tense_data if x['label'] == 1)}")
    logger.info(f"  Future: {sum(1 for x in verb_tense_data if x['label'] == 2)}")

    logger.info(f"Sentiment dataset: {len(sentiment_data)} examples")
    logger.info(f"  Positive: {sum(1 for x in sentiment_data if x['label'] == 0)}")
    logger.info(f"  Negative: {sum(1 for x in sentiment_data if x['label'] == 1)}")
    logger.info(f"  Neutral: {sum(1 for x in sentiment_data if x['label'] == 2)}")

    # Process each layer (skip layer 0 - input embeddings)
    all_results = []

    for layer in range(1, 12):
        logger.info(f"\n{'='*80}")
        logger.info(f"LAYER {layer}")
        logger.info(f"{'='*80}")

        # SKIPPING Plurality task - no signal (separability ratio = 0.0)
        # plurality_acts, plurality_labels = extract_activations(
        #     model, plurality_data, layer, logger
        # )
        # log_diagnostics(plurality_acts, plurality_labels, "Plurality", logger)
        # plurality_pca_results = apply_pca_and_probe(...)
        # plurality_random_results = apply_random_and_probe(...)

        # Process POS task
        logger.info("\nTask: Part of Speech (4-class Classification)")
        logger.info("-" * 80)

        pos_acts, pos_labels = extract_activations(
            model, pos_data, layer, logger
        )

        # Log diagnostics for POS task
        log_diagnostics(pos_acts, pos_labels, "Part of Speech", logger)

        # Method 1: PCA (top 10 components)
        logger.info("\n  Method: PCA (top 10 components)")
        pos_pca_results = apply_pca_and_probe(
            pos_acts,
            pos_labels,
            n_components=args.n_components,
            n_runs=args.n_runs,
            logger=logger
        )

        # Add PCA results
        for run in range(args.n_runs):
            all_results.append({
                'layer': layer,
                'task': 'pos',
                'method': 'pca',
                'run': run,
                'mutual_information': pos_pca_results['mutual_information'][run],
                'accuracy': pos_pca_results['accuracy'][run],
                'f1_score': pos_pca_results['f1_score'][run]
            })

        # Method 2: Random baseline (20 subsets, Gaussian feature sampling)
        logger.info("\n  Method: Random baseline (20 subsets, Gaussian ~ N(384, 50))")
        pos_random_results = apply_random_and_probe(
            pos_acts,
            pos_labels,
            n_subsets=20,
            logger=logger
        )

        # Add random baseline results
        for run in range(20):
            all_results.append({
                'layer': layer,
                'task': 'pos',
                'method': 'random',
                'run': run,
                'mutual_information': pos_random_results['mutual_information'][run],
                'accuracy': pos_random_results['accuracy'][run],
                'f1_score': pos_random_results['f1_score'][run]
            })

        # Process Verb Tense task
        logger.info("\nTask: Verb Tense (3-class Classification)")
        logger.info("-" * 80)

        verb_tense_acts, verb_tense_labels = extract_activations(
            model, verb_tense_data, layer, logger
        )

        log_diagnostics(verb_tense_acts, verb_tense_labels, "Verb Tense", logger)

        # Method 1: PCA
        logger.info("\n  Method: PCA (top 10 components)")
        verb_tense_pca_results = apply_pca_and_probe(
            verb_tense_acts,
            verb_tense_labels,
            n_components=args.n_components,
            n_runs=args.n_runs,
            logger=logger
        )

        for run in range(args.n_runs):
            all_results.append({
                'layer': layer,
                'task': 'verb_tense',
                'method': 'pca',
                'run': run,
                'mutual_information': verb_tense_pca_results['mutual_information'][run],
                'accuracy': verb_tense_pca_results['accuracy'][run],
                'f1_score': verb_tense_pca_results['f1_score'][run]
            })

        # Method 2: Random baseline
        logger.info("\n  Method: Random baseline (20 subsets, Gaussian ~ N(384, 50))")
        verb_tense_random_results = apply_random_and_probe(
            verb_tense_acts,
            verb_tense_labels,
            n_subsets=20,
            logger=logger
        )

        for run in range(20):
            all_results.append({
                'layer': layer,
                'task': 'verb_tense',
                'method': 'random',
                'run': run,
                'mutual_information': verb_tense_random_results['mutual_information'][run],
                'accuracy': verb_tense_random_results['accuracy'][run],
                'f1_score': verb_tense_random_results['f1_score'][run]
            })

        # Process Sentiment task
        logger.info("\nTask: Sentiment (3-class Classification)")
        logger.info("-" * 80)

        sentiment_acts, sentiment_labels = extract_activations(
            model, sentiment_data, layer, logger
        )

        log_diagnostics(sentiment_acts, sentiment_labels, "Sentiment", logger)

        # Method 1: PCA
        logger.info("\n  Method: PCA (top 10 components)")
        sentiment_pca_results = apply_pca_and_probe(
            sentiment_acts,
            sentiment_labels,
            n_components=args.n_components,
            n_runs=args.n_runs,
            logger=logger
        )

        for run in range(args.n_runs):
            all_results.append({
                'layer': layer,
                'task': 'sentiment',
                'method': 'pca',
                'run': run,
                'mutual_information': sentiment_pca_results['mutual_information'][run],
                'accuracy': sentiment_pca_results['accuracy'][run],
                'f1_score': sentiment_pca_results['f1_score'][run]
            })

        # Method 2: Random baseline
        logger.info("\n  Method: Random baseline (20 subsets, Gaussian ~ N(384, 50))")
        sentiment_random_results = apply_random_and_probe(
            sentiment_acts,
            sentiment_labels,
            n_subsets=20,
            logger=logger
        )

        for run in range(20):
            all_results.append({
                'layer': layer,
                'task': 'sentiment',
                'method': 'random',
                'run': run,
                'mutual_information': sentiment_random_results['mutual_information'][run],
                'accuracy': sentiment_random_results['accuracy'][run],
                'f1_score': sentiment_random_results['f1_score'][run]
            })

    # Create results dataframe
    results_df = pd.DataFrame(all_results)

    # Save raw results
    results_path = output_dir / "raw_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"\n{'='*80}")
    logger.info(f"Raw results saved to: {results_path}")

    # Generate plots
    logger.info(f"\n{'='*80}")
    logger.info("GENERATING PLOTS")
    logger.info("="*80)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # SKIPPING Plurality plots - no signal in task
    # plurality_df = results_df[results_df['task'] == 'plurality']
    # plurality_pca_df = plurality_df[plurality_df['method'] == 'pca']
    # plurality_random_df = plurality_df[plurality_df['method'] == 'random']

    # POS plots - PCA vs Random comparison
    pos_df = results_df[results_df['task'] == 'pos']
    pos_pca_df = pos_df[pos_df['method'] == 'pca']
    pos_random_df = pos_df[pos_df['method'] == 'random']

    logger.info("\nPOS - PCA method:")
    create_bar_plot(
        pos_pca_df,
        'mutual_information',
        'Mutual Information',
        'Part of Speech (PCA): Mutual Information Across Layers',
        plots_dir / 'pos_pca_mutual_information.png',
        logger
    )

    create_bar_plot(
        pos_pca_df,
        'accuracy',
        'Accuracy',
        'Part of Speech (PCA): Classification Accuracy Across Layers',
        plots_dir / 'pos_pca_accuracy.png',
        logger
    )

    logger.info("\nPOS - Random baseline:")
    create_bar_plot(
        pos_random_df,
        'mutual_information',
        'Mutual Information',
        'Part of Speech (Random Gaussian): Mutual Information Across Layers',
        plots_dir / 'pos_random_mutual_information.png',
        logger
    )

    create_bar_plot(
        pos_random_df,
        'accuracy',
        'Accuracy',
        'Part of Speech (Random Gaussian): Classification Accuracy Across Layers',
        plots_dir / 'pos_random_accuracy.png',
        logger
    )

    logger.info(f"\nGenerated 4 plots in: {plots_dir} (POS task only - 2 PCA + 2 Random baseline)")

    # Summary statistics
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY STATISTICS")
    logger.info("="*80)

    logger.info("\nPart of Speech Task - PCA (10 components):")
    for layer in range(1, 12):
        layer_df = pos_pca_df[pos_pca_df['layer'] == layer]
        logger.info(
            f"  Layer {layer}: "
            f"MI={layer_df['mutual_information'].mean():.4f} ± {layer_df['mutual_information'].std():.4f}, "
            f"Acc={layer_df['accuracy'].mean():.4f} ± {layer_df['accuracy'].std():.4f}, "
            f"F1={layer_df['f1_score'].mean():.4f} ± {layer_df['f1_score'].std():.4f}"
        )

    logger.info("\nPart of Speech Task - Random Baseline (20 subsets, Gaussian ~ N(384, 50)):")
    for layer in range(1, 12):
        layer_df = pos_random_df[pos_random_df['layer'] == layer]
        logger.info(
            f"  Layer {layer}: "
            f"MI={layer_df['mutual_information'].mean():.4f} ± {layer_df['mutual_information'].std():.4f}, "
            f"Acc={layer_df['accuracy'].mean():.4f} ± {layer_df['accuracy'].std():.4f}, "
            f"F1={layer_df['f1_score'].mean():.4f} ± {layer_df['f1_score'].std():.4f}"
        )

    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
