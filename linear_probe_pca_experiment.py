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

# IMPORTANT: Disable hf_transfer for RunPod compatibility
# Must be set BEFORE any other imports that use HuggingFace Hub
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

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


def create_ner_dataset() -> List[Dict]:
    """
    Create dataset for Named Entity Recognition (NER) task.

    Returns 300 common nouns and 300 proper nouns (600 total).

    Returns:
        List of 600 examples with 'text', 'target_word', and 'label'
        (0=common_noun, 1=proper_noun/named_entity)
    """
    dataset = []

    # Common nouns (300 examples, label=0)
    common_nouns = [
        ("dog", "The dog barked loudly."),
        ("cat", "A cat sat by the window."),
        ("book", "She read a book yesterday."),
        ("table", "The table was wooden."),
        ("chair", "He sat on the chair."),
        ("car", "My car needs repairs."),
        ("house", "They bought a house."),
        ("tree", "The tree was tall."),
        ("flower", "A flower bloomed today."),
        ("computer", "The computer is new."),
        ("phone", "Her phone rang twice."),
        ("door", "The door was open."),
        ("window", "A window broke yesterday."),
        ("street", "The street was quiet."),
        ("city", "The city was busy."),
        ("country", "The country was beautiful."),
        ("river", "A river flowed nearby."),
        ("mountain", "The mountain was steep."),
        ("ocean", "The ocean was calm."),
        ("beach", "The beach was crowded."),
        ("park", "A park was nearby."),
        ("school", "The school was large."),
        ("hospital", "The hospital was modern."),
        ("restaurant", "The restaurant was full."),
        ("store", "A store sold everything."),
        ("market", "The market was open."),
        ("office", "The office was quiet."),
        ("library", "The library was old."),
        ("museum", "A museum displayed art."),
        ("theater", "The theater was dark."),
        ("cinema", "The cinema showed films."),
        ("hotel", "The hotel was expensive."),
        ("airport", "The airport was crowded."),
        ("station", "The station was busy."),
        ("bridge", "A bridge crossed over."),
        ("road", "The road was long."),
        ("path", "The path was narrow."),
        ("garden", "The garden was beautiful."),
        ("field", "A field was empty."),
        ("forest", "The forest was dense."),
        ("desert", "The desert was hot."),
        ("island", "The island was small."),
        ("lake", "The lake was frozen."),
        ("pond", "A pond was nearby."),
        ("valley", "The valley was green."),
        ("hill", "The hill was steep."),
        ("cliff", "A cliff was dangerous."),
        ("cave", "The cave was dark."),
        ("tunnel", "The tunnel was long."),
        ("building", "The building was tall."),
        ("tower", "A tower stood alone."),
        ("castle", "The castle was ancient."),
        ("palace", "The palace was grand."),
        ("temple", "The temple was sacred."),
        ("church", "The church was old."),
        ("mosque", "The mosque was beautiful."),
        ("shrine", "A shrine was hidden."),
        ("monument", "The monument was impressive."),
        ("statue", "The statue was bronze."),
        ("fountain", "A fountain flowed continuously."),
        ("bench", "The bench was wooden."),
        ("lamp", "The lamp was bright."),
        ("clock", "A clock ticked loudly."),
        ("mirror", "The mirror was cracked."),
        ("picture", "The picture was colorful."),
        ("painting", "A painting hung there."),
        ("photograph", "The photograph was old."),
        ("map", "The map was detailed."),
        ("calendar", "A calendar showed dates."),
        ("newspaper", "The newspaper was fresh."),
        ("magazine", "The magazine was interesting."),
        ("letter", "A letter arrived today."),
        ("envelope", "The envelope was sealed."),
        ("package", "The package was heavy."),
        ("box", "A box was empty."),
        ("bag", "The bag was full."),
        ("basket", "The basket was woven."),
        ("bottle", "A bottle was broken."),
        ("cup", "The cup was clean."),
        ("glass", "The glass was empty."),
        ("plate", "A plate was dirty."),
        ("bowl", "The bowl was full."),
        ("spoon", "The spoon was silver."),
        ("fork", "A fork was missing."),
        ("knife", "The knife was sharp."),
        ("pot", "The pot was hot."),
        ("pan", "A pan was heavy."),
        ("stove", "The stove was old."),
        ("oven", "The oven was hot."),
        ("refrigerator", "The refrigerator was empty."),
        ("sink", "A sink was clogged."),
        ("toilet", "The toilet was clean."),
        ("shower", "The shower was broken."),
        ("bath", "A bath was relaxing."),
        ("towel", "The towel was soft."),
        ("soap", "The soap smelled nice."),
        ("shampoo", "A shampoo bottle fell."),
        ("toothbrush", "The toothbrush was new."),
        ("toothpaste", "The toothpaste was mint."),
        ("comb", "A comb was missing."),
        ("brush", "The brush was old."),
        ("razor", "The razor was sharp."),
        ("scissors", "A scissors was rusty."),
        ("needle", "The needle was thin."),
        ("thread", "The thread was strong."),
        ("fabric", "A fabric was soft."),
        ("cloth", "The cloth was clean."),
        ("blanket", "The blanket was warm."),
        ("pillow", "A pillow was comfortable."),
        ("sheet", "The sheet was white."),
        ("mattress", "The mattress was firm."),
        ("bed", "A bed was unmade."),
        ("sofa", "The sofa was comfortable."),
        ("couch", "The couch was old."),
        ("armchair", "An armchair was cozy."),
        ("desk", "The desk was messy."),
        ("shelf", "The shelf was full."),
        ("cabinet", "A cabinet was locked."),
        ("drawer", "The drawer was stuck."),
        ("wardrobe", "The wardrobe was large."),
        ("closet", "A closet was organized."),
        ("hanger", "The hanger was metal."),
        ("carpet", "The carpet was soft."),
        ("rug", "A rug was colorful."),
        ("curtain", "The curtain was closed."),
        ("blind", "The blind was broken."),
        ("wall", "A wall was painted."),
        ("ceiling", "The ceiling was high."),
        ("floor", "The floor was clean."),
        ("roof", "The roof leaked rain."),
        ("chimney", "A chimney smoked heavily."),
        ("stairs", "The stairs were steep."),
        ("elevator", "The elevator was slow."),
        ("escalator", "An escalator was broken."),
        ("fence", "The fence was tall."),
        ("gate", "The gate was locked."),
        ("wall", "A wall surrounded it."),
        ("hedge", "The hedge was trimmed."),
        ("lawn", "The lawn was green."),
        ("grass", "A grass was wet."),
        ("weed", "The weed was stubborn."),
        ("plant", "The plant was healthy."),
        ("bush", "A bush was flowering."),
        ("shrub", "The shrub was dense."),
        ("vine", "The vine climbed high."),
        ("leaf", "A leaf fell down."),
        ("branch", "The branch was broken."),
        ("trunk", "The trunk was thick."),
        ("root", "A root was exposed."),
        ("seed", "The seed was tiny."),
        ("fruit", "The fruit was ripe."),
        ("vegetable", "A vegetable was fresh."),
        ("grain", "The grain was stored."),
        ("wheat", "The wheat was golden."),
        ("rice", "A rice was cooked."),
        ("corn", "The corn was sweet."),
        ("potato", "The potato was baked."),
        ("tomato", "A tomato was red."),
        ("carrot", "The carrot was orange."),
        ("onion", "The onion was strong."),
        ("garlic", "A garlic was fresh."),
        ("pepper", "The pepper was spicy."),
        ("salt", "The salt was white."),
        ("sugar", "A sugar was sweet."),
        ("flour", "The flour was fine."),
        ("bread", "The bread was fresh."),
        ("cake", "A cake was delicious."),
        ("cookie", "The cookie was crunchy."),
        ("pie", "The pie was warm."),
        ("pizza", "A pizza was hot."),
        ("sandwich", "The sandwich was tasty."),
        ("burger", "The burger was juicy."),
        ("chicken", "A chicken was roasted."),
        ("beef", "The beef was tender."),
        ("pork", "The pork was lean."),
        ("fish", "A fish was fresh."),
        ("egg", "The egg was boiled."),
        ("milk", "The milk was cold."),
        ("cheese", "A cheese was aged."),
        ("butter", "The butter was soft."),
        ("cream", "The cream was thick."),
        ("yogurt", "A yogurt was tasty."),
        ("ice", "The ice was melting."),
        ("water", "The water was clear."),
        ("juice", "A juice was fresh."),
        ("coffee", "The coffee was hot."),
        ("tea", "The tea was warm."),
        ("wine", "A wine was expensive."),
        ("beer", "The beer was cold."),
        ("soda", "The soda was fizzy."),
        ("candy", "A candy was sweet."),
        ("chocolate", "The chocolate was dark."),
        ("toy", "The toy was broken."),
        ("game", "A game was fun."),
        ("puzzle", "The puzzle was hard."),
        ("ball", "The ball was round."),
        ("bat", "A bat was wooden."),
        ("glove", "The glove was leather."),
        ("shoe", "The shoe was worn."),
        ("boot", "A boot was muddy."),
        ("sock", "The sock was clean."),
        ("shirt", "The shirt was white."),
        ("pants", "A pants was torn."),
        ("dress", "The dress was beautiful."),
        ("skirt", "The skirt was short."),
        ("coat", "A coat was warm."),
        ("jacket", "The jacket was leather."),
        ("sweater", "The sweater was wool."),
        ("hat", "A hat was stylish."),
        ("cap", "The cap was red."),
        ("scarf", "The scarf was long."),
        ("tie", "A tie was silk."),
        ("belt", "The belt was leather."),
        ("watch", "The watch was expensive."),
        ("ring", "A ring was gold."),
        ("necklace", "The necklace was silver."),
        ("bracelet", "The bracelet was pretty."),
        ("earring", "An earring was missing."),
        ("wallet", "The wallet was empty."),
        ("purse", "The purse was full."),
        ("backpack", "A backpack was heavy."),
        ("suitcase", "The suitcase was large."),
        ("umbrella", "The umbrella was broken."),
        ("cane", "A cane was wooden."),
        ("wheelchair", "The wheelchair was modern."),
        ("crutch", "The crutch was helpful."),
        ("bandage", "A bandage was clean."),
        ("medicine", "The medicine was bitter."),
        ("pill", "The pill was small."),
        ("tablet", "A tablet was white."),
        ("syringe", "The syringe was sterile."),
        ("thermometer", "The thermometer was digital."),
        ("stethoscope", "A stethoscope was useful."),
        ("microscope", "The microscope was powerful."),
        ("telescope", "The telescope was large."),
        ("camera", "A camera was expensive."),
        ("lens", "The lens was clean."),
        ("film", "The film was old."),
        ("video", "A video was interesting."),
        ("television", "The television was large."),
        ("radio", "The radio was old."),
        ("speaker", "A speaker was loud."),
        ("microphone", "The microphone was sensitive."),
        ("headphone", "The headphone was comfortable."),
        ("keyboard", "A keyboard was mechanical."),
        ("mouse", "The mouse was wireless."),
        ("monitor", "The monitor was wide."),
        ("printer", "A printer was broken."),
        ("scanner", "The scanner was fast."),
        ("cable", "The cable was long."),
        ("wire", "A wire was exposed."),
        ("battery", "The battery was dead."),
        ("charger", "The charger was missing."),
        ("plug", "A plug was loose."),
        ("socket", "The socket was empty."),
        ("switch", "The switch was broken."),
        ("button", "A button was missing."),
        ("lever", "The lever was stiff."),
        ("handle", "The handle was broken."),
        ("knob", "A knob was loose."),
        ("wheel", "The wheel was round."),
        ("tire", "The tire was flat."),
        ("engine", "An engine was loud."),
        ("motor", "The motor was powerful."),
        ("machine", "The machine was old."),
        ("tool", "A tool was missing."),
        ("hammer", "The hammer was heavy."),
        ("screwdriver", "The screwdriver was useful."),
        ("wrench", "A wrench was adjustable."),
        ("pliers", "The pliers were rusty."),
        ("saw", "The saw was sharp."),
        ("drill", "A drill was powerful."),
        ("nail", "The nail was bent."),
        ("screw", "The screw was loose."),
        ("bolt", "A bolt was tight."),
        ("nut", "The nut was missing."),
    ]

    for word, text in common_nouns:
        dataset.append({
            'text': text,
            'target_word': word,
            'label': 0
        })

    # Proper nouns / Named entities (300 examples, label=1)
    proper_nouns = [
        ("John", "John went to school."),
        ("Mary", "Mary loves reading books."),
        ("David", "David plays the guitar."),
        ("Sarah", "Sarah is a doctor."),
        ("Michael", "Michael runs every day."),
        ("Emily", "Emily speaks three languages."),
        ("James", "James works in finance."),
        ("Emma", "Emma enjoys painting."),
        ("Robert", "Robert lives in Texas."),
        ("Linda", "Linda teaches mathematics."),
        ("William", "William loves baseball."),
        ("Lisa", "Lisa studies chemistry."),
        ("Richard", "Richard builds houses."),
        ("Jennifer", "Jennifer writes novels."),
        ("Thomas", "Thomas plays piano."),
        ("Susan", "Susan works hard."),
        ("Charles", "Charles enjoys hiking."),
        ("Jessica", "Jessica is very smart."),
        ("Daniel", "Daniel runs marathons."),
        ("Karen", "Karen loves animals."),
        ("Matthew", "Matthew studies history."),
        ("Nancy", "Nancy teaches English."),
        ("Joseph", "Joseph is a chef."),
        ("Betty", "Betty sings beautifully."),
        ("Christopher", "Christopher plays tennis."),
        ("Margaret", "Margaret loves gardening."),
        ("Steven", "Steven works remotely."),
        ("Dorothy", "Dorothy is a nurse."),
        ("Andrew", "Andrew enjoys fishing."),
        ("Sandra", "Sandra is very kind."),
        ("Kevin", "Kevin loves sports."),
        ("Ashley", "Ashley studies law."),
        ("Jason", "Jason is a musician."),
        ("Kimberly", "Kimberly enjoys cooking."),
        ("Brian", "Brian plays basketball."),
        ("Donna", "Donna is a teacher."),
        ("George", "George loves reading."),
        ("Carol", "Carol enjoys knitting."),
        ("Ryan", "Ryan is very athletic."),
        ("Michelle", "Michelle studies biology."),
        ("London", "London is a city."),
        ("Paris", "Paris is beautiful."),
        ("Tokyo", "Tokyo is very busy."),
        ("Berlin", "Berlin has history."),
        ("Rome", "Rome is ancient."),
        ("Madrid", "Madrid is warm."),
        ("Athens", "Athens is historic."),
        ("Dublin", "Dublin is charming."),
        ("Prague", "Prague is beautiful."),
        ("Vienna", "Vienna is elegant."),
        ("Venice", "Venice has canals."),
        ("Barcelona", "Barcelona is vibrant."),
        ("Amsterdam", "Amsterdam has bikes."),
        ("Brussels", "Brussels is central."),
        ("Stockholm", "Stockholm is clean."),
        ("Copenhagen", "Copenhagen is modern."),
        ("Oslo", "Oslo is expensive."),
        ("Helsinki", "Helsinki is cold."),
        ("Warsaw", "Warsaw is rebuilding."),
        ("Budapest", "Budapest is stunning."),
        ("Moscow", "Moscow is large."),
        ("Beijing", "Beijing is crowded."),
        ("Shanghai", "Shanghai is modern."),
        ("Seoul", "Seoul is technological."),
        ("Bangkok", "Bangkok is busy."),
        ("Singapore", "Singapore is clean."),
        ("Sydney", "Sydney is beautiful."),
        ("Melbourne", "Melbourne is cultural."),
        ("Toronto", "Toronto is diverse."),
        ("Vancouver", "Vancouver is scenic."),
        ("Montreal", "Montreal is bilingual."),
        ("Chicago", "Chicago is windy."),
        ("Boston", "Boston is historic."),
        ("Seattle", "Seattle is rainy."),
        ("Miami", "Miami is sunny."),
        ("Atlanta", "Atlanta is growing."),
        ("Denver", "Denver is high."),
        ("Portland", "Portland is green."),
        ("Phoenix", "Phoenix is hot."),
        ("Dallas", "Dallas is sprawling."),
        ("Houston", "Houston is large."),
        ("Philadelphia", "Philadelphia is historic."),
        ("Detroit", "Detroit is recovering."),
        ("Memphis", "Memphis has music."),
        ("Nashville", "Nashville is musical."),
        ("Austin", "Austin is quirky."),
        ("SanFrancisco", "SanFrancisco is hilly."),
        ("LosAngeles", "LosAngeles is sprawling."),
        ("NewYork", "NewYork is busy."),
        ("England", "England is historic."),
        ("France", "France is beautiful."),
        ("Germany", "Germany is efficient."),
        ("Italy", "Italy has art."),
        ("Spain", "Spain is warm."),
        ("Greece", "Greece is ancient."),
        ("Portugal", "Portugal is sunny."),
        ("Ireland", "Ireland is green."),
        ("Scotland", "Scotland is rugged."),
        ("Wales", "Wales has mountains."),
        ("Netherlands", "Netherlands is flat."),
        ("Belgium", "Belgium has chocolate."),
        ("Switzerland", "Switzerland is expensive."),
        ("Austria", "Austria is alpine."),
        ("Poland", "Poland is historic."),
        ("Hungary", "Hungary is central."),
        ("Russia", "Russia is huge."),
        ("China", "China is ancient."),
        ("Japan", "Japan is modern."),
        ("Korea", "Korea is divided."),
        ("India", "India is diverse."),
        ("Thailand", "Thailand is tropical."),
        ("Vietnam", "Vietnam is beautiful."),
        ("Indonesia", "Indonesia is archipelagic."),
        ("Malaysia", "Malaysia is multicultural."),
        ("Australia", "Australia is vast."),
        ("Canada", "Canada is cold."),
        ("Mexico", "Mexico is warm."),
        ("Brazil", "Brazil is large."),
        ("Argentina", "Argentina is southern."),
        ("Chile", "Chile is long."),
        ("Peru", "Peru is mountainous."),
        ("Egypt", "Egypt is ancient."),
        ("Morocco", "Morocco is colorful."),
        ("Kenya", "Kenya has wildlife."),
        ("SouthAfrica", "SouthAfrica is diverse."),
        ("Monday", "Monday is busy."),
        ("Tuesday", "Tuesday is productive."),
        ("Wednesday", "Wednesday is midweek."),
        ("Thursday", "Thursday is almost done."),
        ("Friday", "Friday is exciting."),
        ("Saturday", "Saturday is relaxing."),
        ("Sunday", "Sunday is restful."),
        ("January", "January is cold."),
        ("February", "February is short."),
        ("March", "March is windy."),
        ("April", "April has showers."),
        ("May", "May has flowers."),
        ("June", "June is sunny."),
        ("July", "July is hot."),
        ("August", "August is warm."),
        ("September", "September is beautiful."),
        ("October", "October has colors."),
        ("November", "November is chilly."),
        ("December", "December is festive."),
        ("Amazon", "Amazon sells everything."),
        ("Google", "Google knows everything."),
        ("Apple", "Apple makes phones."),
        ("Microsoft", "Microsoft makes software."),
        ("Facebook", "Facebook connects people."),
        ("Twitter", "Twitter shares news."),
        ("Netflix", "Netflix streams shows."),
        ("Spotify", "Spotify plays music."),
        ("Tesla", "Tesla makes cars."),
        ("Toyota", "Toyota is reliable."),
        ("Ford", "Ford makes trucks."),
        ("Honda", "Honda is efficient."),
        ("BMW", "BMW is luxurious."),
        ("Mercedes", "Mercedes is expensive."),
        ("Volkswagen", "Volkswagen is German."),
        ("Nike", "Nike makes shoes."),
        ("Adidas", "Adidas sponsors athletes."),
        ("Puma", "Puma is sporty."),
        ("Coca-Cola", "Coca-Cola is sweet."),
        ("Pepsi", "Pepsi is refreshing."),
        ("Starbucks", "Starbucks sells coffee."),
        ("McDonald's", "McDonald's serves burgers."),
        ("Disney", "Disney makes movies."),
        ("Warner", "Warner produces films."),
        ("Sony", "Sony makes electronics."),
        ("Samsung", "Samsung is innovative."),
        ("Intel", "Intel makes chips."),
        ("AMD", "AMD competes well."),
        ("IBM", "IBM is historic."),
        ("Oracle", "Oracle manages databases."),
        ("Cisco", "Cisco networks systems."),
        ("Harvard", "Harvard is prestigious."),
        ("Stanford", "Stanford is innovative."),
        ("MIT", "MIT is technical."),
        ("Yale", "Yale is historic."),
        ("Princeton", "Princeton is exclusive."),
        ("Oxford", "Oxford is ancient."),
        ("Cambridge", "Cambridge is renowned."),
        ("Columbia", "Columbia is urban."),
        ("Berkeley", "Berkeley is liberal."),
        ("UCLA", "UCLA is large."),
        ("Christmas", "Christmas is festive."),
        ("Easter", "Easter has eggs."),
        ("Halloween", "Halloween is spooky."),
        ("Thanksgiving", "Thanksgiving has turkey."),
        ("Valentine", "Valentine has love."),
        ("Patrick", "Patrick is Irish."),
        ("Independence", "Independence is celebrated."),
        ("Memorial", "Memorial honors soldiers."),
        ("Labor", "Labor recognizes workers."),
        ("Veteran", "Veteran honors service."),
        ("Shakespeare", "Shakespeare wrote plays."),
        ("Mozart", "Mozart composed music."),
        ("Beethoven", "Beethoven was deaf."),
        ("DaVinci", "DaVinci painted masterpieces."),
        ("Picasso", "Picasso was innovative."),
        ("Einstein", "Einstein was brilliant."),
        ("Newton", "Newton discovered gravity."),
        ("Darwin", "Darwin studied evolution."),
        ("Galileo", "Galileo studied stars."),
        ("Copernicus", "Copernicus was revolutionary."),
        ("Columbus", "Columbus sailed west."),
        ("Napoleon", "Napoleon conquered Europe."),
        ("Caesar", "Caesar ruled Rome."),
        ("Cleopatra", "Cleopatra ruled Egypt."),
        ("Alexander", "Alexander conquered lands."),
        ("Washington", "Washington was first."),
        ("Lincoln", "Lincoln freed slaves."),
        ("Roosevelt", "Roosevelt led well."),
        ("Kennedy", "Kennedy inspired people."),
        ("Churchill", "Churchill was resolute."),
        ("Gandhi", "Gandhi was peaceful."),
        ("Mandela", "Mandela fought apartheid."),
        ("King", "King had dreams."),
        ("Everest", "Everest is tall."),
        ("Kilimanjaro", "Kilimanjaro is African."),
        ("Fuji", "Fuji is iconic."),
        ("Alps", "Alps are snowy."),
        ("Himalayas", "Himalayas are massive."),
        ("Rockies", "Rockies are rugged."),
        ("Andes", "Andes are long."),
        ("Sahara", "Sahara is vast."),
        ("Amazon", "Amazon is dense."),
        ("Nile", "Nile is long."),
        ("Mississippi", "Mississippi is wide."),
        ("Thames", "Thames flows through."),
        ("Seine", "Seine is romantic."),
        ("Danube", "Danube is historic."),
        ("Rhine", "Rhine is important."),
        ("Pacific", "Pacific is huge."),
        ("Atlantic", "Atlantic is wide."),
        ("Indian", "Indian is warm."),
        ("Arctic", "Arctic is frozen."),
        ("Antarctic", "Antarctic is cold."),
        ("Mediterranean", "Mediterranean is beautiful."),
        ("Caribbean", "Caribbean is tropical."),
        ("BlackSea", "BlackSea is historic."),
        ("RedSea", "RedSea is warm."),
        ("Jupiter", "Jupiter is largest."),
        ("Mars", "Mars is red."),
        ("Venus", "Venus is bright."),
        ("Saturn", "Saturn has rings."),
        ("Mercury", "Mercury is closest."),
        ("Neptune", "Neptune is distant."),
        ("Uranus", "Uranus is tilted."),
        ("Pluto", "Pluto was demoted."),
        ("Earth", "Earth sustains life."),
        ("Sun", "Sun provides energy."),
        ("Moon", "Moon orbits Earth."),
    ]

    for word, text in proper_nouns:
        dataset.append({
            'text': text,
            'target_word': word,
            'label': 1
        })

    return dataset


def create_word_length_dataset() -> List[Dict]:
    """
    Create dataset for word length prediction task.

    Returns 200 examples for each length category (600 total).

    Returns:
        List of 600 examples with 'text', 'target_word', and 'label'
        (0=short (3-5 letters), 1=medium (6-8 letters), 2=long (9+ letters))
    """
    dataset = []

    # Short words: 3-5 letters (200 examples, label=0)
    short_words = [
        ("cat", "The cat is here."),
        ("dog", "A dog runs fast."),
        ("bird", "The bird flies high."),
        ("fish", "A fish swims well."),
        ("tree", "The tree is tall."),
        ("car", "A car drives by."),
        ("book", "The book is good."),
        ("pen", "A pen writes well."),
        ("cup", "The cup is full."),
        ("hat", "A hat fits well."),
        ("bag", "The bag is heavy."),
        ("box", "A box is empty."),
        ("key", "The key is lost."),
        ("door", "A door is open."),
        ("hand", "The hand is warm."),
        ("foot", "A foot is sore."),
        ("head", "The head is clear."),
        ("face", "A face is kind."),
        ("eye", "The eye sees far."),
        ("ear", "An ear hears well."),
        ("nose", "The nose smells good."),
        ("hair", "A hair is long."),
        ("neck", "The neck is stiff."),
        ("arm", "An arm is strong."),
        ("leg", "The leg is tired."),
        ("back", "A back hurts now."),
        ("skin", "The skin is soft."),
        ("bone", "A bone is broken."),
        ("mind", "The mind is clear."),
        ("heart", "A heart beats fast."),
        ("soul", "The soul is pure."),
        ("life", "A life is short."),
        ("time", "The time is now."),
        ("day", "A day passes by."),
        ("week", "The week is long."),
        ("year", "A year flies by."),
        ("hour", "The hour is late."),
        ("moon", "A moon is full."),
        ("star", "The star is bright."),
        ("sun", "A sun shines warm."),
        ("rain", "The rain falls hard."),
        ("snow", "A snow is white."),
        ("wind", "The wind blows cold."),
        ("fire", "A fire burns hot."),
        ("water", "The water is cold."),
        ("earth", "An earth is round."),
        ("stone", "The stone is hard."),
        ("metal", "A metal is cold."),
        ("glass", "The glass is clear."),
        ("wood", "The wood is solid."),
        ("paper", "A paper is thin."),
        ("cloth", "The cloth is soft."),
        ("rope", "A rope is strong."),
        ("wire", "The wire is thin."),
        ("chain", "A chain is heavy."),
        ("belt", "The belt is tight."),
        ("ring", "A ring is gold."),
        ("coin", "The coin is old."),
        ("bill", "A bill is due."),
        ("card", "The card is valid."),
        ("stamp", "A stamp is rare."),
        ("sign", "The sign is clear."),
        ("flag", "A flag waves high."),
        ("map", "The map is old."),
        ("plan", "A plan is ready."),
        ("goal", "The goal is near."),
        ("dream", "A dream is vivid."),
        ("hope", "The hope is strong."),
        ("fear", "A fear is real."),
        ("love", "The love is true."),
        ("hate", "A hate is wrong."),
        ("joy", "The joy is real."),
        ("pain", "A pain is sharp."),
        ("anger", "The anger is hot."),
        ("peace", "A peace is calm."),
        ("war", "The war is over."),
        ("friend", "A friend is true."),
        ("enemy", "The enemy is near."),
        ("group", "A group is large."),
        ("team", "The team wins games."),
        ("crowd", "A crowd gathers now."),
        ("man", "The man is tall."),
        ("woman", "A woman is smart."),
        ("child", "The child is young."),
        ("baby", "A baby cries loud."),
        ("boy", "The boy runs fast."),
        ("girl", "A girl sings well."),
        ("king", "The king rules well."),
        ("queen", "A queen is wise."),
        ("lord", "The lord is fair."),
        ("lady", "A lady is kind."),
        ("sir", "The sir is polite."),
        ("boss", "A boss is strict."),
        ("worker", "The worker is tired."),
        ("farmer", "A farmer plants crops."),
        ("doctor", "The doctor helps sick."),
        ("nurse", "A nurse cares much."),
        ("cook", "The cook makes food."),
        ("baker", "A baker bakes bread."),
        ("driver", "The driver is safe."),
        ("pilot", "A pilot flies high."),
        ("sailor", "The sailor is brave."),
        ("soldier", "A soldier stands guard."),
        ("police", "The police patrol streets."),
        ("judge", "A judge is fair."),
        ("lawyer", "The lawyer argues well."),
        ("artist", "An artist paints well."),
        ("singer", "The singer has voice."),
        ("dancer", "A dancer moves well."),
        ("actor", "The actor is good."),
        ("writer", "A writer tells tales."),
        ("poet", "The poet writes verse."),
        ("player", "A player is skilled."),
        ("coach", "The coach trains hard."),
        ("fan", "A fan cheers loud."),
        ("hero", "The hero saves lives."),
        ("saint", "A saint is holy."),
        ("fool", "The fool acts dumb."),
        ("wise", "A wise person knows."),
        ("brave", "The brave stand firm."),
        ("smart", "A smart person learns."),
        ("kind", "The kind help others."),
        ("cruel", "A cruel person hurts."),
        ("happy", "The happy smile wide."),
        ("sad", "A sad person cries."),
        ("calm", "The calm stay still."),
        ("wild", "A wild runs free."),
        ("tame", "The tame obey well."),
        ("hot", "A hot burns skin."),
        ("cold", "The cold chills bones."),
        ("warm", "A warm feels nice."),
        ("cool", "The cool is nice."),
        ("wet", "A wet is damp."),
        ("dry", "The dry is crisp."),
        ("clean", "A clean is pure."),
        ("dirty", "The dirty is messy."),
        ("new", "A new is fresh."),
        ("old", "The old is worn."),
        ("young", "A young is fresh."),
        ("fast", "The fast moves quick."),
        ("slow", "A slow takes time."),
        ("big", "The big is huge."),
        ("small", "A small is tiny."),
        ("tall", "The tall reaches high."),
        ("short", "A short is brief."),
        ("wide", "The wide spans far."),
        ("thin", "A thin is narrow."),
        ("thick", "The thick is dense."),
        ("light", "A light is bright."),
        ("dark", "The dark is deep."),
        ("soft", "A soft is gentle."),
        ("hard", "The hard is firm."),
        ("easy", "An easy is simple."),
        ("hard", "The hard is tough."),
        ("good", "A good is nice."),
        ("bad", "The bad is wrong."),
        ("right", "The right is correct."),
        ("wrong", "A wrong is error."),
        ("true", "The true is real."),
        ("false", "A false is fake."),
        ("real", "The real is true."),
        ("fake", "A fake is false."),
        ("full", "The full is complete."),
        ("empty", "An empty is void."),
        ("open", "The open is wide."),
        ("shut", "A shut is closed."),
        ("far", "The far is distant."),
        ("near", "A near is close."),
        ("high", "The high is up."),
        ("low", "A low is down."),
        ("rich", "The rich has money."),
        ("poor", "A poor lacks funds."),
        ("safe", "The safe is secure."),
        ("danger", "A danger is risky."),
        ("health", "The health is good."),
        ("sick", "A sick needs help."),
        ("strong", "The strong lifts much."),
        ("weak", "A weak tires fast."),
        ("loud", "The loud is noisy."),
        ("quiet", "A quiet is silent."),
        ("first", "The first is ahead."),
        ("last", "A last is behind."),
        ("best", "The best is top."),
        ("worst", "A worst is bottom."),
        ("more", "The more is extra."),
        ("less", "A less is fewer."),
        ("most", "The most is maximum."),
        ("least", "A least is minimum."),
    ]

    for word, text in short_words:
        dataset.append({
            'text': text,
            'target_word': word,
            'label': 0
        })

    # Medium words: 6-8 letters (200 examples, label=1)
    medium_words = [
        ("computer", "The computer is fast."),
        ("keyboard", "A keyboard types well."),
        ("monitor", "The monitor is bright."),
        ("printer", "A printer jams often."),
        ("scanner", "The scanner works well."),
        ("speaker", "A speaker plays loud."),
        ("camera", "The camera takes photos."),
        ("picture", "A picture is pretty."),
        ("painting", "The painting is old."),
        ("drawing", "A drawing is nice."),
        ("building", "The building is tall."),
        ("window", "A window is clean."),
        ("ceiling", "The ceiling is high."),
        ("kitchen", "A kitchen is warm."),
        ("bedroom", "The bedroom is cozy."),
        ("bathroom", "A bathroom is clean."),
        ("garden", "The garden is green."),
        ("forest", "A forest is dense."),
        ("mountain", "The mountain is steep."),
        ("valley", "A valley is green."),
        ("desert", "The desert is hot."),
        ("island", "An island is small."),
        ("ocean", "The ocean is vast."),
        ("river", "A river flows fast."),
        ("lake", "The lake is calm."),
        ("stream", "A stream is clear."),
        ("bridge", "The bridge is strong."),
        ("tunnel", "A tunnel is dark."),
        ("highway", "The highway is busy."),
        ("street", "A street is quiet."),
        ("avenue", "The avenue is wide."),
        ("road", "A road is long."),
        ("path", "The path is narrow."),
        ("trail", "A trail is steep."),
        ("airport", "The airport is busy."),
        ("station", "A station is crowded."),
        ("harbor", "The harbor is peaceful."),
        ("market", "A market is lively."),
        ("store", "The store is open."),
        ("shop", "A shop sells goods."),
        ("office", "The office is quiet."),
        ("factory", "A factory makes things."),
        ("warehouse", "The warehouse is full."),
        ("library", "A library is silent."),
        ("museum", "The museum is old."),
        ("theater", "A theater shows plays."),
        ("cinema", "The cinema is dark."),
        ("restaurant", "A restaurant serves food."),
        ("cafe", "The cafe is cozy."),
        ("hotel", "A hotel is expensive."),
        ("hospital", "The hospital is busy."),
        ("school", "A school teaches kids."),
        ("college", "The college is large."),
        ("church", "A church is peaceful."),
        ("temple", "The temple is sacred."),
        ("mosque", "A mosque is beautiful."),
        ("castle", "The castle is ancient."),
        ("palace", "A palace is grand."),
        ("tower", "The tower is tall."),
        ("statue", "A statue is bronze."),
        ("monument", "The monument is large."),
        ("fountain", "A fountain is flowing."),
        ("bench", "The bench is wooden."),
        ("table", "A table is sturdy."),
        ("chair", "The chair is comfortable."),
        ("couch", "A couch is soft."),
        ("desk", "The desk is messy."),
        ("shelf", "A shelf is full."),
        ("cabinet", "The cabinet is locked."),
        ("drawer", "A drawer is stuck."),
        ("closet", "The closet is organized."),
        ("wardrobe", "A wardrobe is large."),
        ("mirror", "The mirror is cracked."),
        ("lamp", "A lamp is bright."),
        ("candle", "The candle is burning."),
        ("curtain", "A curtain is closed."),
        ("carpet", "The carpet is soft."),
        ("blanket", "A blanket is warm."),
        ("pillow", "The pillow is fluffy."),
        ("mattress", "A mattress is firm."),
        ("towel", "The towel is dry."),
        ("soap", "A soap smells good."),
        ("shampoo", "The shampoo is new."),
        ("toothbrush", "A toothbrush is clean."),
        ("toothpaste", "The toothpaste is minty."),
        ("razor", "A razor is sharp."),
        ("scissors", "The scissors are dull."),
        ("needle", "A needle is thin."),
        ("thread", "The thread is strong."),
        ("fabric", "A fabric is soft."),
        ("leather", "The leather is tough."),
        ("cotton", "A cotton is natural."),
        ("wool", "The wool is warm."),
        ("silk", "A silk is smooth."),
        ("plastic", "The plastic is cheap."),
        ("metal", "A metal is strong."),
        ("steel", "The steel is hard."),
        ("iron", "An iron is heavy."),
        ("copper", "The copper is shiny."),
        ("silver", "A silver is valuable."),
        ("gold", "The gold is precious."),
        ("diamond", "A diamond is rare."),
        ("pearl", "The pearl is white."),
        ("ruby", "A ruby is red."),
        ("emerald", "The emerald is green."),
        ("sapphire", "A sapphire is blue."),
        ("crystal", "The crystal is clear."),
        ("marble", "A marble is smooth."),
        ("granite", "The granite is hard."),
        ("concrete", "A concrete is solid."),
        ("brick", "The brick is red."),
        ("stone", "A stone is heavy."),
        ("sand", "The sand is fine."),
        ("clay", "A clay is soft."),
        ("mud", "The mud is wet."),
        ("dirt", "The dirt is brown."),
        ("dust", "A dust is everywhere."),
        ("powder", "The powder is fine."),
        ("liquid", "A liquid flows easily."),
        ("solid", "The solid is firm."),
        ("gas", "A gas expands fast."),
        ("vapor", "The vapor is hot."),
        ("smoke", "A smoke is thick."),
        ("flame", "The flame is hot."),
        ("spark", "A spark is bright."),
        ("light", "The light is dim."),
        ("shadow", "A shadow is dark."),
        ("color", "The color is vivid."),
        ("shade", "A shade is cool."),
        ("tone", "The tone is warm."),
        ("sound", "A sound is loud."),
        ("noise", "The noise is annoying."),
        ("voice", "A voice is clear."),
        ("music", "The music is beautiful."),
        ("song", "A song is catchy."),
        ("melody", "The melody is sweet."),
        ("rhythm", "A rhythm is steady."),
        ("beat", "The beat is strong."),
        ("tempo", "A tempo is fast."),
        ("pitch", "The pitch is high."),
        ("volume", "A volume is loud."),
        ("silence", "The silence is peaceful."),
        ("smell", "A smell is strong."),
        ("scent", "The scent is pleasant."),
        ("odor", "An odor is bad."),
        ("aroma", "The aroma is lovely."),
        ("perfume", "A perfume is expensive."),
        ("taste", "The taste is sweet."),
        ("flavor", "A flavor is rich."),
        ("texture", "The texture is smooth."),
        ("feeling", "A feeling is strong."),
        ("emotion", "The emotion is deep."),
        ("thought", "A thought is fleeting."),
        ("idea", "The idea is brilliant."),
        ("concept", "A concept is abstract."),
        ("theory", "The theory is complex."),
        ("fact", "A fact is true."),
        ("truth", "The truth is clear."),
        ("lie", "A lie is wrong."),
        ("story", "The story is long."),
        ("tale", "A tale is old."),
        ("legend", "The legend is famous."),
        ("myth", "A myth is false."),
        ("fable", "The fable teaches well."),
        ("joke", "A joke is funny."),
        ("riddle", "The riddle is hard."),
        ("puzzle", "A puzzle is tricky."),
        ("mystery", "The mystery is deep."),
        ("secret", "A secret is hidden."),
        ("clue", "The clue is helpful."),
        ("answer", "An answer is correct."),
        ("question", "The question is hard."),
        ("problem", "A problem is tough."),
        ("solution", "The solution is simple."),
        ("method", "A method is effective."),
        ("system", "The system works well."),
        ("process", "A process takes time."),
        ("procedure", "The procedure is complex."),
        ("technique", "A technique is skillful."),
        ("skill", "The skill is learned."),
        ("talent", "A talent is natural."),
        ("ability", "The ability is rare."),
        ("power", "A power is strong."),
        ("force", "The force is mighty."),
        ("energy", "An energy is high."),
        ("strength", "The strength is great."),
        ("weakness", "A weakness is shown."),
        ("advantage", "The advantage is clear."),
    ]

    for word, text in medium_words:
        dataset.append({
            'text': text,
            'target_word': word,
            'label': 1
        })

    # Long words: 9+ letters (200 examples, label=2)
    long_words = [
        ("wonderful", "This is wonderful news."),
        ("beautiful", "The view is beautiful."),
        ("incredible", "That's incredible work."),
        ("fantastic", "The show was fantastic."),
        ("magnificent", "A magnificent sight indeed."),
        ("spectacular", "The performance was spectacular."),
        ("extraordinary", "An extraordinary achievement today."),
        ("remarkable", "This is remarkable progress."),
        ("impressive", "Very impressive results shown."),
        ("outstanding", "Their work is outstanding."),
        ("excellent", "The quality is excellent."),
        ("brilliant", "A brilliant idea emerged."),
        ("marvelous", "The outcome is marvelous."),
        ("splendid", "What a splendid evening."),
        ("delightful", "This is delightful indeed."),
        ("charming", "How charming this place."),
        ("enchanting", "An enchanting atmosphere here."),
        ("captivating", "The story is captivating."),
        ("fascinating", "This topic is fascinating."),
        ("interesting", "Very interesting findings shown."),
        ("important", "This is important news."),
        ("significant", "A significant discovery made."),
        ("essential", "This is essential information."),
        ("necessary", "Necessary steps were taken."),
        ("critical", "A critical moment arrived."),
        ("crucial", "This is crucial timing."),
        ("vital", "Vital signs are stable."),
        ("fundamental", "The fundamental principles apply."),
        ("principal", "The principal reason given."),
        ("primary", "The primary concern addressed."),
        ("secondary", "A secondary effect observed."),
        ("tertiary", "The tertiary stage reached."),
        ("elementary", "Elementary concepts taught first."),
        ("advanced", "Advanced techniques were used."),
        ("sophisticated", "A sophisticated approach taken."),
        ("complicated", "The process is complicated."),
        ("complex", "A complex system exists."),
        ("simple", "The solution is simple."),
        ("straightforward", "A straightforward answer given."),
        ("difficult", "This is difficult work."),
        ("challenging", "A challenging task ahead."),
        ("demanding", "The job is demanding."),
        ("strenuous", "Strenuous effort was required."),
        ("exhausting", "The work was exhausting."),
        ("tiring", "A tiring day passed."),
        ("relaxing", "The massage was relaxing."),
        ("refreshing", "A refreshing drink served."),
        ("invigorating", "An invigorating walk taken."),
        ("energizing", "The workout was energizing."),
        ("stimulating", "A stimulating conversation held."),
        ("exciting", "The game was exciting."),
        ("thrilling", "A thrilling adventure began."),
        ("exhilarating", "An exhilarating experience had."),
        ("breathtaking", "The view was breathtaking."),
        ("astonishing", "The news was astonishing."),
        ("astounding", "An astounding discovery made."),
        ("amazing", "The results were amazing."),
        ("surprising", "A surprising turn occurred."),
        ("unexpected", "An unexpected visitor came."),
        ("anticipated", "The anticipated results appeared."),
        ("predicted", "The predicted outcome happened."),
        ("forecasted", "Forecasted weather was accurate."),
        ("estimated", "The estimated time arrived."),
        ("calculated", "A calculated risk was taken."),
        ("measured", "Measured responses were given."),
        ("evaluated", "The data was evaluated."),
        ("assessed", "The situation was assessed."),
        ("analyzed", "The results were analyzed."),
        ("examined", "The evidence was examined."),
        ("investigated", "The case was investigated."),
        ("researched", "The topic was researched."),
        ("studied", "The subject was studied."),
        ("observed", "The behavior was observed."),
        ("monitored", "The progress was monitored."),
        ("supervised", "The project was supervised."),
        ("managed", "The team was managed."),
        ("directed", "The film was directed."),
        ("controlled", "The experiment was controlled."),
        ("regulated", "The industry is regulated."),
        ("governed", "The country is governed."),
        ("administered", "The test was administered."),
        ("organized", "The event was organized."),
        ("arranged", "The meeting was arranged."),
        ("coordinated", "The efforts were coordinated."),
        ("integrated", "The systems were integrated."),
        ("combined", "The ingredients were combined."),
        ("merged", "The companies were merged."),
        ("unified", "The teams were unified."),
        ("connected", "The devices were connected."),
        ("linked", "The pages were linked."),
        ("associated", "The concepts were associated."),
        ("related", "The topics are related."),
        ("correlated", "The variables were correlated."),
        ("corresponding", "Corresponding results were found."),
        ("equivalent", "The values are equivalent."),
        ("comparable", "The results are comparable."),
        ("similar", "The patterns are similar."),
        ("different", "The approaches are different."),
        ("distinct", "The categories are distinct."),
        ("separate", "The issues are separate."),
        ("independent", "The variables are independent."),
        ("dependent", "The outcome is dependent."),
        ("conditional", "The offer is conditional."),
        ("unconditional", "The love is unconditional."),
        ("absolute", "The power is absolute."),
        ("relative", "The position is relative."),
        ("comparative", "A comparative study done."),
        ("superlative", "The superlative form used."),
        ("positive", "The feedback was positive."),
        ("negative", "The result was negative."),
        ("neutral", "The stance is neutral."),
        ("objective", "An objective view taken."),
        ("subjective", "A subjective opinion given."),
        ("personal", "This is personal business."),
        ("individual", "Each individual case examined."),
        ("collective", "The collective decision made."),
        ("universal", "A universal truth exists."),
        ("general", "The general consensus reached."),
        ("specific", "Specific details were provided."),
        ("particular", "This particular case matters."),
        ("special", "A special occasion celebrated."),
        ("ordinary", "An ordinary day passed."),
        ("common", "The common practice followed."),
        ("rare", "A rare opportunity arose."),
        ("unique", "This is unique design."),
        ("unusual", "An unusual event occurred."),
        ("normal", "The normal procedure followed."),
        ("standard", "The standard method used."),
        ("typical", "A typical response given."),
        ("atypical", "An atypical case appeared."),
        ("irregular", "An irregular pattern shown."),
        ("regular", "The regular schedule kept."),
        ("consistent", "The quality is consistent."),
        ("inconsistent", "The results were inconsistent."),
        ("constant", "A constant effort maintained."),
        ("variable", "The variable factors considered."),
        ("stable", "The condition is stable."),
        ("unstable", "The structure is unstable."),
        ("balanced", "A balanced approach taken."),
        ("unbalanced", "The equation is unbalanced."),
        ("symmetrical", "The design is symmetrical."),
        ("asymmetrical", "The pattern is asymmetrical."),
        ("proportional", "The response was proportional."),
        ("disproportionate", "The reaction was disproportionate."),
        ("appropriate", "The response was appropriate."),
        ("inappropriate", "The comment was inappropriate."),
        ("suitable", "The candidate is suitable."),
        ("unsuitable", "The location is unsuitable."),
        ("acceptable", "The offer is acceptable."),
        ("unacceptable", "The behavior is unacceptable."),
        ("satisfactory", "The work is satisfactory."),
        ("unsatisfactory", "The performance was unsatisfactory."),
        ("adequate", "The resources are adequate."),
        ("inadequate", "The preparation was inadequate."),
        ("sufficient", "The evidence is sufficient."),
        ("insufficient", "The data is insufficient."),
        ("excessive", "The spending was excessive."),
        ("moderate", "The temperature is moderate."),
        ("minimal", "The damage was minimal."),
        ("maximal", "The effort was maximal."),
        ("optimal", "The conditions are optimal."),
        ("suboptimal", "The results were suboptimal."),
        ("efficient", "The system is efficient."),
        ("inefficient", "The process is inefficient."),
        ("effective", "The treatment is effective."),
        ("ineffective", "The method was ineffective."),
        ("productive", "The meeting was productive."),
        ("unproductive", "The discussion was unproductive."),
        ("successful", "The project was successful."),
        ("unsuccessful", "The attempt was unsuccessful."),
        ("favorable", "The conditions are favorable."),
        ("unfavorable", "The weather is unfavorable."),
        ("advantageous", "The position is advantageous."),
        ("disadvantageous", "The timing is disadvantageous."),
        ("beneficial", "The change is beneficial."),
        ("detrimental", "The effect is detrimental."),
        ("positive", "The impact is positive."),
        ("negative", "The consequence is negative."),
        ("constructive", "The criticism was constructive."),
        ("destructive", "The behavior is destructive."),
        ("creative", "The solution is creative."),
        ("innovative", "The approach is innovative."),
        ("traditional", "The method is traditional."),
        ("conventional", "The treatment is conventional."),
        ("unconventional", "The strategy is unconventional."),
        ("alternative", "An alternative route exists."),
        ("mainstream", "The opinion is mainstream."),
        ("experimental", "The treatment is experimental."),
        ("theoretical", "The framework is theoretical."),
        ("practical", "The advice is practical."),
        ("impractical", "The plan is impractical."),
        ("realistic", "The goal is realistic."),
        ("unrealistic", "The expectation is unrealistic."),
        ("reasonable", "The request is reasonable."),
        ("unreasonable", "The demand is unreasonable."),
    ]

    for word, text in long_words:
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
    logger: logging.Logger,
    hook: str = "resid_post"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract activations at target token positions for all examples.

    Args:
        model: HookedTransformer model
        examples: List of examples with 'text', 'target_word', 'label'
        layer: Layer index to extract from
        logger: Logger instance
        hook: Hook point type (e.g., "resid_post", "resid_pre")

    Returns:
        Tuple of (activations, labels) as numpy arrays
        activations: (n_examples, d_model)
        labels: (n_examples,)
    """
    hook_name = f"blocks.{layer}.hook_{hook}"
    activations_list = []
    labels_list = []
    token_positions = []  # Track all token positions

    # DEBUG: Print first few examples
    if layer == 1:
        logger.info(f"  [DEBUG] First 3 examples:")
        for i in range(min(3, len(examples))):
            logger.info(f"    {i}: text='{examples[i]['text']}', target='{examples[i]['target_word']}', label={examples[i]['label']}")

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
            # DEBUG: Track token positions for first layer
            if layer == 1 and len(activations_list) < 5:
                logger.info(f"  [DEBUG] Example {len(activations_list)}: target_pos={target_pos}, text='{text}'")

            # Track all positions for statistics
            token_positions.append(target_pos)
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

    # DEBUG: Token position statistics
    if layer == 1:
        unique_positions, position_counts = np.unique(token_positions, return_counts=True)
        logger.info(f"  [DEBUG] Token position distribution: {dict(zip(unique_positions, position_counts))}")
        logger.info(f"  [DEBUG] Position range: {min(token_positions)} to {max(token_positions)}")
        logger.info(f"  [DEBUG] Most common position: {unique_positions[np.argmax(position_counts)]} ({max(position_counts)}/{len(token_positions)} examples)")

    # DEBUG: Check if activations are identical across examples
    if len(activations) > 1:
        act_diff = np.abs(activations[0] - activations[1]).max()
        logger.info(f"  [DEBUG] Max diff between first two activations: {act_diff:.6f}")
        logger.info(f"  [DEBUG] First 5 labels: {labels[:5]}")
        logger.info(f"  [DEBUG] First activation mean: {activations[0].mean():.6f}, std: {activations[0].std():.6f}")
        logger.info(f"  [DEBUG] Second activation mean: {activations[1].mean():.6f}, std: {activations[1].std():.6f}")

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

        # DEBUG: Check prediction distribution
        if run == 0 and logger:
            unique_preds, pred_counts = np.unique(predictions, return_counts=True)
            logger.info(f"  [DEBUG] Prediction distribution: {dict(zip(unique_preds, pred_counts))}")
            logger.info(f"  [DEBUG] First 10 predictions: {predictions[:10]}")
            logger.info(f"  [DEBUG] First 10 labels: {labels[:10]}")

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
    n_features: int = None,  # Ignored
    n_subsets: int = 3,
    logger: logging.Logger = None,
    random_mean: int = None,  # Ignored if use_fixed_size=True or use_uniform_size=True
    random_std: int = None,  # Ignored if use_fixed_size=True or use_uniform_size=True
    use_fixed_size: bool = False,
    fixed_size_ratio: int = 20,
    use_uniform_size: bool = False  # If True, uniformly sample subset size from [1, d_model]
) -> Dict:
    """
    Sample random feature subsets and train probes (baseline comparison).

    Args:
        activations: (n_examples, d_model) activation matrix
        labels: (n_examples,) label array
        n_features: Ignored (kept for backward compatibility)
        n_subsets: Number of random subsets to try (default: 3)
        logger: Logger instance
        random_mean: Mean for Gaussian sampling (ignored if use_fixed_size=True or use_uniform_size=True)
        random_std: Std for Gaussian sampling (ignored if use_fixed_size=True or use_uniform_size=True)
        use_fixed_size: If True, use fixed subset size = d_model / fixed_size_ratio
        fixed_size_ratio: Ratio for fixed size (default: 20, gives d_model/20)
        use_uniform_size: If True, uniformly sample subset size from [1, d_model] for each subset

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

    d_model = standardized_activations.shape[1]

    mi_scores = []
    accuracy_scores = []
    f1_scores = []
    n_features_list = []

    # Track used subsets to ensure uniqueness
    used_subsets = set()

    for subset_idx in range(n_subsets):
        np.random.seed(42 + subset_idx)  # Reproducible

        if use_uniform_size:
            # Uniformly sample subset size from [1, d_model]
            n_features_sample = np.random.randint(1, d_model + 1)
        elif use_fixed_size:
            # Fixed size = d_model / fixed_size_ratio
            n_features_sample = d_model // fixed_size_ratio
        else:
            # Gaussian sampling
            if random_mean is None:
                mean_features = d_model // 20
            else:
                mean_features = random_mean

            if random_std is None:
                std_features = 5
            else:
                std_features = random_std

            n_features_sample = int(np.random.normal(mean_features, std_features))
            n_features_sample = max(10, min(d_model, n_features_sample))

        n_features_list.append(n_features_sample)

        # Sample unique features (no duplicates across subsets)
        max_attempts = 1000
        for attempt in range(max_attempts):
            selected_features = np.random.choice(d_model, size=n_features_sample, replace=False)
            feature_tuple = tuple(sorted(selected_features))

            if feature_tuple not in used_subsets:
                used_subsets.add(feature_tuple)
                break
        else:
            # If we couldn't find a unique subset after max_attempts, just use the last one
            if logger:
                logger.warning(f"Could not find unique subset after {max_attempts} attempts for subset {subset_idx}")

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
        if use_uniform_size:
            logger.info(f"  Random baseline ({n_subsets} subsets, uniform size from [1, d_model]):")
        elif use_fixed_size:
            logger.info(f"  Random baseline ({n_subsets} subsets, fixed size = d_model/{fixed_size_ratio}):")
        else:
            logger.info(f"  Random baseline ({n_subsets} subsets, Gaussian features ~ N({random_mean}, {random_std})):")
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
        description="Linear probe PCA experiment on transformer models"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (optional)"
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
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2-small",
        help="Model name (default: gpt2-small)"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="1-11",
        help="Layers to probe (format: '1-11' or '1,5,10,15')"
    )
    parser.add_argument(
        "--hook",
        type=str,
        default="resid_post",
        help="Hook point type (default: resid_post)"
    )
    args = parser.parse_args()

    # Load config if provided
    config = {}
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    # Override defaults with config values (config takes precedence over CLI defaults)
    model_name = config.get("model_name", args.model_name)
    hook = config.get("hook", args.hook)
    output_dir_str = config.get("output_dir", args.output_dir)
    n_components = config.get("n_components", args.n_components)
    n_runs = config.get("n_runs", args.n_runs)
    seed = config.get("seed", args.seed)

    # Random baseline configuration
    n_subsets = config.get("n_subsets", 3)  # Default 3 subsets
    random_mean = config.get("random_mean", None)  # If None, will use d_model/20
    random_std = config.get("random_std", None)  # If None, will use 5
    use_fixed_size = config.get("use_fixed_size", False)  # If True, use fixed subset size
    fixed_size_ratio = config.get("fixed_size_ratio", 20)  # Default: d_model/20
    use_uniform_size = config.get("use_uniform_size", False)  # If True, uniformly sample subset size from [1, d_model]

    # Parse layers
    if "layers" in config:
        layers_config = config["layers"]
        if isinstance(layers_config, list):
            layers = layers_config
        else:
            # Handle string format like "1-11"
            if "-" in str(layers_config):
                start, end = map(int, str(layers_config).split("-"))
                layers = list(range(start, end + 1))
            else:
                layers = [int(layers_config)]
    else:
        # Parse from CLI arg
        if "-" in args.layers:
            start, end = map(int, args.layers.split("-"))
            layers = list(range(start, end + 1))
        elif "," in args.layers:
            layers = [int(x.strip()) for x in args.layers.split(",")]
        else:
            layers = [int(args.layers)]

    # Setup output directory
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    logger.info("="*80)
    logger.info("LINEAR PROBE PCA EXPERIMENT")
    logger.info("="*80)
    if args.config:
        logger.info(f"Config file: {args.config}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Hook point: {hook}")
    logger.info(f"Layers: {layers}")
    logger.info(f"Output directory: {output_dir}")
    if use_uniform_size:
        logger.info(f"Random subsets: {n_subsets} subsets, uniform size from [1, d_model]")
    elif use_fixed_size:
        logger.info(f"Random subsets: {n_subsets} subsets, fixed size = d_model/{fixed_size_ratio}")
    else:
        logger.info(f"Random subsets: {n_subsets} subsets, N({random_mean if random_mean else 'd_model/20'}, {random_std if random_std else 5})")
    logger.info(f"Random seed: {seed}")

    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set device
    device_config = config.get("device", "auto")
    if device_config == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_config)
    logger.info(f"Using device: {device}")

    # Load model
    logger.info("\n" + "="*80)
    logger.info("LOADING MODEL")
    logger.info("="*80)

    model_loader = ModelLoader(
        model_name=model_name,
        hook=hook,
        layers=layers,
        device=device,
        logger=logger
    )

    model = model_loader.load_model()

    # Verify model is on correct device
    logger.info(f"Model device: {next(model.parameters()).device}")

    # Create datasets
    logger.info("\n" + "="*80)
    logger.info("CREATING DATASETS")
    logger.info("="*80)

    # SKIPPING plurality dataset - no signal (separability ratio = 0.0)
    # plurality_data = create_plurality_dataset()
    pos_data = create_pos_dataset()
    # Import diverse datasets with varied sentence structures
    from fixed_datasets import (
        create_ner_dataset_diverse,
        create_word_length_dataset_diverse,
        create_sentiment_dataset_diverse
    )
    ner_data = create_ner_dataset_diverse()
    word_length_data = create_word_length_dataset_diverse()
    sentiment_data = create_sentiment_dataset_diverse()

    logger.info(f"POS dataset: {len(pos_data)} examples")
    logger.info(f"  Nouns: {sum(1 for x in pos_data if x['label'] == 0)}")
    logger.info(f"  Verbs: {sum(1 for x in pos_data if x['label'] == 1)}")
    logger.info(f"  Adjectives: {sum(1 for x in pos_data if x['label'] == 2)}")
    logger.info(f"  Adverbs: {sum(1 for x in pos_data if x['label'] == 3)}")

    logger.info(f"NER dataset: {len(ner_data)} examples")
    logger.info(f"  Common nouns: {sum(1 for x in ner_data if x['label'] == 0)}")
    logger.info(f"  Proper nouns/Named entities: {sum(1 for x in ner_data if x['label'] == 1)}")

    logger.info(f"Word Length dataset: {len(word_length_data)} examples")
    logger.info(f"  Short (3-5 letters): {sum(1 for x in word_length_data if x['label'] == 0)}")
    logger.info(f"  Medium (6-8 letters): {sum(1 for x in word_length_data if x['label'] == 1)}")
    logger.info(f"  Long (9+ letters): {sum(1 for x in word_length_data if x['label'] == 2)}")

    logger.info(f"Sentiment dataset: {len(sentiment_data)} examples")
    logger.info(f"  Positive: {sum(1 for x in sentiment_data if x['label'] == 0)}")
    logger.info(f"  Negative: {sum(1 for x in sentiment_data if x['label'] == 1)}")
    logger.info(f"  Neutral: {sum(1 for x in sentiment_data if x['label'] == 2)}")

    # Process each layer
    all_results = []

    for layer in layers:
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
            model, pos_data, layer, logger, hook
        )

        # Log diagnostics for POS task
        log_diagnostics(pos_acts, pos_labels, "Part of Speech", logger)

        # Random baseline
        if use_uniform_size:
            logger.info(f"\n  Method: Random baseline ({n_subsets} subsets, uniform size from [1, d_model])")
        elif use_fixed_size:
            logger.info(f"\n  Method: Random baseline ({n_subsets} subsets, fixed size = d_model/{fixed_size_ratio})")
        else:
            logger.info(f"\n  Method: Random baseline ({n_subsets} subsets, Gaussian ~ N({random_mean if random_mean else 'd_model/20'}, {random_std if random_std else 5}))")
        pos_random_results = apply_random_and_probe(
            pos_acts,
            pos_labels,
            n_subsets=n_subsets,
            logger=logger,
            random_mean=random_mean,
            random_std=random_std,
            use_fixed_size=use_fixed_size,
            fixed_size_ratio=fixed_size_ratio,
            use_uniform_size=use_uniform_size
        )

        # Add random baseline results
        for run in range(n_subsets):
            all_results.append({
                'layer': layer,
                'task': 'pos',
                'method': 'random',
                'run': run,
                'mutual_information': pos_random_results['mutual_information'][run],
                'accuracy': pos_random_results['accuracy'][run],
                'f1_score': pos_random_results['f1_score'][run]
            })

        # Process NER task
        logger.info("\nTask: NER - Named Entity Recognition (Binary Classification)")
        logger.info("-" * 80)

        ner_acts, ner_labels = extract_activations(
            model, ner_data, layer, logger, hook
        )

        log_diagnostics(ner_acts, ner_labels, "NER (Named Entity Recognition)", logger)

        # Method 2: Random baseline
        if use_uniform_size:
            logger.info(f"\n  Method: Random baseline ({n_subsets} subsets, uniform size from [1, d_model])")
        elif use_fixed_size:
            logger.info(f"\n  Method: Random baseline ({n_subsets} subsets, fixed size = d_model/{fixed_size_ratio})")
        else:
            logger.info(f"\n  Method: Random baseline ({n_subsets} subsets, Gaussian ~ N({random_mean if random_mean else 'd_model/20'}, {random_std if random_std else 5}))")
        ner_random_results = apply_random_and_probe(
            ner_acts,
            ner_labels,
            n_subsets=n_subsets,
            logger=logger,
            random_mean=random_mean,
            random_std=random_std,
            use_fixed_size=use_fixed_size,
            fixed_size_ratio=fixed_size_ratio,
            use_uniform_size=use_uniform_size
        )

        for run in range(n_subsets):
            all_results.append({
                'layer': layer,
                'task': 'ner',
                'method': 'random',
                'run': run,
                'mutual_information': ner_random_results['mutual_information'][run],
                'accuracy': ner_random_results['accuracy'][run],
                'f1_score': ner_random_results['f1_score'][run]
            })

        # Process Word Length task
        logger.info("\nTask: Word Length (3-class Classification)")
        logger.info("-" * 80)

        word_length_acts, word_length_labels = extract_activations(
            model, word_length_data, layer, logger, hook
        )

        log_diagnostics(word_length_acts, word_length_labels, "Word Length", logger)

        # Method 2: Random baseline
        if use_uniform_size:
            logger.info(f"\n  Method: Random baseline ({n_subsets} subsets, uniform size from [1, d_model])")
        elif use_fixed_size:
            logger.info(f"\n  Method: Random baseline ({n_subsets} subsets, fixed size = d_model/{fixed_size_ratio})")
        else:
            logger.info(f"\n  Method: Random baseline ({n_subsets} subsets, Gaussian ~ N({random_mean if random_mean else 'd_model/20'}, {random_std if random_std else 5}))")
        word_length_random_results = apply_random_and_probe(
            word_length_acts,
            word_length_labels,
            n_subsets=n_subsets,
            logger=logger,
            random_mean=random_mean,
            random_std=random_std,
            use_fixed_size=use_fixed_size,
            fixed_size_ratio=fixed_size_ratio,
            use_uniform_size=use_uniform_size
        )

        for run in range(n_subsets):
            all_results.append({
                'layer': layer,
                'task': 'word_length',
                'method': 'random',
                'run': run,
                'mutual_information': word_length_random_results['mutual_information'][run],
                'accuracy': word_length_random_results['accuracy'][run],
                'f1_score': word_length_random_results['f1_score'][run]
            })

        # Process Sentiment task
        logger.info("\nTask: Sentiment (3-class Classification)")
        logger.info("-" * 80)

        sentiment_acts, sentiment_labels = extract_activations(
            model, sentiment_data, layer, logger, hook
        )

        log_diagnostics(sentiment_acts, sentiment_labels, "Sentiment", logger)

        # Method 2: Random baseline
        if use_uniform_size:
            logger.info(f"\n  Method: Random baseline ({n_subsets} subsets, uniform size from [1, d_model])")
        elif use_fixed_size:
            logger.info(f"\n  Method: Random baseline ({n_subsets} subsets, fixed size = d_model/{fixed_size_ratio})")
        else:
            logger.info(f"\n  Method: Random baseline ({n_subsets} subsets, Gaussian ~ N({random_mean if random_mean else 'd_model/20'}, {random_std if random_std else 5}))")
        sentiment_random_results = apply_random_and_probe(
            sentiment_acts,
            sentiment_labels,
            n_subsets=n_subsets,
            logger=logger,
            random_mean=random_mean,
            random_std=random_std,
            use_fixed_size=use_fixed_size,
            fixed_size_ratio=fixed_size_ratio,
            use_uniform_size=use_uniform_size
        )

        for run in range(n_subsets):
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

    # POS plots - Random baseline only
    pos_df = results_df[results_df['task'] == 'pos']
    pos_random_df = pos_df[pos_df['method'] == 'random']

    # Determine baseline method for plot titles
    if use_uniform_size:
        baseline_method = "Random Uniform-Size"
    elif use_fixed_size:
        baseline_method = "Random Fixed-Size"
    else:
        baseline_method = "Random Gaussian"

    logger.info("\nPOS - Random baseline:")
    create_bar_plot(
        pos_random_df,
        'mutual_information',
        'Mutual Information',
        f'Part of Speech ({baseline_method}): Mutual Information Across Layers',
        plots_dir / 'pos_random_mutual_information.png',
        logger
    )

    create_bar_plot(
        pos_random_df,
        'accuracy',
        'Accuracy',
        f'Part of Speech ({baseline_method}): Classification Accuracy Across Layers',
        plots_dir / 'pos_random_accuracy.png',
        logger
    )

    # NER plots - Random baseline only
    ner_df = results_df[results_df['task'] == 'ner']
    ner_random_df = ner_df[ner_df['method'] == 'random']

    logger.info("\nNER (Named Entity Recognition) - Random baseline:")
    create_bar_plot(
        ner_random_df,
        'mutual_information',
        'Mutual Information',
        f'NER ({baseline_method}): Mutual Information Across Layers',
        plots_dir / 'ner_random_mutual_information.png',
        logger
    )

    create_bar_plot(
        ner_random_df,
        'accuracy',
        'Accuracy',
        f'NER ({baseline_method}): Classification Accuracy Across Layers',
        plots_dir / 'ner_random_accuracy.png',
        logger
    )

    # Word Length plots - Random baseline only
    word_length_df = results_df[results_df['task'] == 'word_length']
    word_length_random_df = word_length_df[word_length_df['method'] == 'random']

    logger.info("\nWord Length - Random baseline:")
    create_bar_plot(
        word_length_random_df,
        'mutual_information',
        'Mutual Information',
        f'Word Length ({baseline_method}): Mutual Information Across Layers',
        plots_dir / 'word_length_random_mutual_information.png',
        logger
    )

    create_bar_plot(
        word_length_random_df,
        'accuracy',
        'Accuracy',
        f'Word Length ({baseline_method}): Classification Accuracy Across Layers',
        plots_dir / 'word_length_random_accuracy.png',
        logger
    )

    # Sentiment plots - Random baseline only
    sentiment_df = results_df[results_df['task'] == 'sentiment']
    sentiment_random_df = sentiment_df[sentiment_df['method'] == 'random']

    logger.info("\nSentiment - Random baseline:")
    create_bar_plot(
        sentiment_random_df,
        'mutual_information',
        'Mutual Information',
        f'Sentiment ({baseline_method}): Mutual Information Across Layers',
        plots_dir / 'sentiment_random_mutual_information.png',
        logger
    )

    create_bar_plot(
        sentiment_random_df,
        'accuracy',
        'Accuracy',
        f'Sentiment ({baseline_method}): Classification Accuracy Across Layers',
        plots_dir / 'sentiment_random_accuracy.png',
        logger
    )

    logger.info(f"\nGenerated 8 plots in: {plots_dir} (4 tasks × 1 method × 2 metrics)")

    # Summary statistics
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY STATISTICS")
    logger.info("="*80)

    if use_uniform_size:
        baseline_desc = f"{n_subsets} subsets, uniform size from [1, d_model]"
    elif use_fixed_size:
        baseline_desc = f"{n_subsets} subsets, fixed size = d_model/{fixed_size_ratio}"
    else:
        baseline_desc = f"{n_subsets} subsets, Gaussian ~ N({random_mean if random_mean else 'd_model/20'}, {random_std if random_std else 5})"

    logger.info(f"\nPart of Speech Task - Random Baseline ({baseline_desc}):")
    for layer in layers:
        layer_df = pos_random_df[pos_random_df['layer'] == layer]
        logger.info(
            f"  Layer {layer}: "
            f"MI={layer_df['mutual_information'].mean():.4f} ± {layer_df['mutual_information'].std():.4f}, "
            f"Acc={layer_df['accuracy'].mean():.4f} ± {layer_df['accuracy'].std():.4f}, "
            f"F1={layer_df['f1_score'].mean():.4f} ± {layer_df['f1_score'].std():.4f}"
        )

    logger.info(f"\nNER Task - Random Baseline ({baseline_desc}):")
    for layer in layers:
        layer_df = ner_random_df[ner_random_df['layer'] == layer]
        logger.info(
            f"  Layer {layer}: "
            f"MI={layer_df['mutual_information'].mean():.4f} ± {layer_df['mutual_information'].std():.4f}, "
            f"Acc={layer_df['accuracy'].mean():.4f} ± {layer_df['accuracy'].std():.4f}, "
            f"F1={layer_df['f1_score'].mean():.4f} ± {layer_df['f1_score'].std():.4f}"
        )

    logger.info(f"\nWord Length Task - Random Baseline ({baseline_desc}):")
    for layer in layers:
        layer_df = word_length_random_df[word_length_random_df['layer'] == layer]
        logger.info(
            f"  Layer {layer}: "
            f"MI={layer_df['mutual_information'].mean():.4f} ± {layer_df['mutual_information'].std():.4f}, "
            f"Acc={layer_df['accuracy'].mean():.4f} ± {layer_df['accuracy'].std():.4f}, "
            f"F1={layer_df['f1_score'].mean():.4f} ± {layer_df['f1_score'].std():.4f}"
        )

    logger.info(f"\nSentiment Task - Random Baseline ({baseline_desc}):")
    for layer in layers:
        layer_df = sentiment_random_df[sentiment_random_df['layer'] == layer]
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
