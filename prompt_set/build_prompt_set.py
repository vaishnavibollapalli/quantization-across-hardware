#!/usr/bin/env python3
"""
Build the fixed prompt set (150 prompts) for "Comparing Quantization Across Hardware".

Outputs (written next to this script):
  perf_prompts.jsonl        50 prompts for latency / throughput / memory / energy / thermal runs
                            (12 flagged "core": true for the repeated 3-run measurements)
  perf/<ID>.txt             each perf prompt's user text as a plain file (for llama-cli -f)
  quality_prompts.jsonl     100 held-out prompts with answer keys for FP16 / INT8 / INT4 quality
  passage/report_a.txt      long-context document A (original, fictional)
  passage/report_b.txt      long-context document B (original, fictional)
  rubric.json               1-5 scale for the 6 manually scored items
  token_counts_<tag>.csv    prompt length per prompt (exact if a GGUF + llama-tokenize is given)

Usage:
  python3 build_prompt_set.py                                   # estimated token counts
  python3 build_prompt_set.py --model models/qwen2.5-3b-q8_0.gguf \
          --tokenizer-bin ~/llama.cpp/build/bin/llama-tokenize  # exact counts for that model
Run the exact count once per model family (Phi-3, Qwen2.5, Llama): tokenizers differ,
so the same text gives different token counts per model.
"""
import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SYSTEM = "You are a helpful, concise assistant."

# ======================================================================================
# Long-context documents: 2 original fictional reports, 8 blocks of ~250 tokens each.
# Written for this benchmark, so no model has memorized them and there is no copyright issue.
# ======================================================================================
DOC_A_TITLE = "Kestrel Valley Community Microgrid: 2025 Annual Operations Report"
DOC_A = [
    # A1 - Overview
    "The Kestrel Valley Community Microgrid completed its fourth full year of operation in 2025. "
    "The system serves 1,240 households and 38 businesses across the town and the surrounding farm roads. "
    "Generation comes from a 4.2 MW solar array spread over three sites, a 1.5 MW wind installation on "
    "Harrow Ridge, and a lithium iron phosphate battery system. During 2025, the microgrid supplied 71 percent "
    "of the town's total electricity demand, up from 58 percent in 2024. The operations team attributes most "
    "of this improvement to the second battery bank, commissioned in March 2025, which raised total storage "
    "from 3 MWh to 6 MWh. With more storage available, surplus midday solar energy could be held for the "
    "evening peak instead of being curtailed or exported at low prices. Purchases from the regional grid fell "
    "by roughly one third compared with the previous year. The microgrid remained connected to the regional "
    "utility throughout the year and continued to import power during long stretches of cloudy, low-wind "
    "weather. This report summarizes generation, storage, reliability, costs, community programs, and the "
    "plans approved by the cooperative board for 2026.",
    # A2 - Seasonal generation
    "Generation followed a familiar seasonal pattern. January and February were the weakest months for solar "
    "output, as short days and repeated snowfall reduced production; crews cleared the two most accessible "
    "arrays after each major storm, but the hillside array at Pell Orchard often stayed covered for several "
    "days. Wind partly offset the shortfall: the Harrow Ridge turbines produced their highest monthly total in "
    "February, at 498 MWh. Solar output climbed steadily through spring and peaked in June at 812 MWh, the "
    "highest single-month figure in the system's history. July and August were slightly lower because of haze "
    "from distant wildfires, which reduced irradiance on several afternoons. Autumn brought a gradual decline "
    "in solar production and a return of stronger winds in late October and November. Across the full year, "
    "solar contributed about 64 percent of local generation and wind about 36 percent. The team noted that "
    "the combination of the two sources remains valuable, since the windiest weeks rarely coincide with the "
    "sunniest ones. Forecasting improved as well: day-ahead solar forecasts were within ten percent of actual "
    "output on 83 percent of days, compared with 71 percent in 2024.",
    # A3 - Battery operations
    "The expanded battery system completed 287 full equivalent charge cycles in 2025. Typical daily operation "
    "involved charging from surplus solar between late morning and mid-afternoon, then discharging between "
    "5 p.m. and 10 p.m., when household demand is highest. Annual capacity testing in December measured a 2.1 "
    "percent loss of usable capacity in the original battery bank, which is within the manufacturer's expected "
    "range for a system of its age. The newer bank showed no measurable loss. The most significant battery "
    "incident occurred in July, when a cooling fan in the original enclosure failed during a heat wave. "
    "Internal temperatures rose above the recommended limit, and the battery management system automatically "
    "reduced the discharge rate by half until a replacement fan was installed two days later. No cells were "
    "damaged. Following the incident, the team added remote temperature alerts that notify on-call staff by "
    "text message, and it now keeps two spare fans on site. Staff also began a quarterly inspection of air "
    "filters, which had become clogged with pollen and dust more quickly than expected during the dry summer "
    "months.",
    # A4 - Reliability
    "The microgrid recorded three unplanned outages in 2025, compared with five in 2024. The first, in "
    "January, lasted 22 minutes and was caused by ice loading on a distribution line near the school. The "
    "second, in May, lasted 48 minutes after a vehicle struck a utility pole on Route 9. The longest outage "
    "occurred on August 14, when a lightning strike damaged equipment at the main inverter station and left "
    "part of the town without power for 3 hours and 40 minutes. In each case, the microgrid's islanding "
    "controls worked as designed for the sections that were not physically damaged: those areas disconnected "
    "from the regional grid and continued operating on local solar and battery power. During a separate "
    "regional blackout in September, which lasted more than six hours, the entire town remained powered by "
    "the microgrid alone. Residents and local businesses widely cited this event as the clearest "
    "demonstration of the system's value. The team has since installed additional surge protection at the "
    "inverter station and is reviewing the grounding design with an outside engineering firm.",
    # A5 - Costs
    "Average residential electricity rates in Kestrel Valley were 13.8 cents per kilowatt-hour in 2025, "
    "compared with a regional average of 16.1 cents. Commercial rates averaged 11.9 cents. The cooperative's "
    "total operating expenses were 2.3 million dollars, of which maintenance accounted for about 18 percent "
    "and debt service on the second battery bank accounted for about 27 percent. Revenue exceeded expenses by "
    "a modest margin, and the board placed the surplus in a reserve fund for equipment replacement. Lower "
    "purchases from the regional grid were the largest single source of savings. However, insurance premiums "
    "rose by 12 percent after the lightning incident, and the cost of replacement inverter parts was higher "
    "than budgeted. The board decided to keep rates unchanged for 2026 rather than lowering them, in order to "
    "rebuild the reserve fund more quickly. A rate review is scheduled for the end of 2026 once the effects of "
    "the planned storage expansion are known. The cooperative also received a state grant that covered part "
    "of the cost of the forecasting software upgrade completed in the spring.",
    # A6 - Community programs
    "Community programs continued to grow alongside the physical system. The heat pump rebate program, which "
    "offers up to 2,500 dollars toward replacing oil or propane heating, was used by 214 households in 2025, "
    "bringing the program's three-year total to 506 homes. Because heat pumps increase winter electricity "
    "demand, the operations team now includes expected heat pump adoption in its seasonal load forecasts. The "
    "cooperative also installed 12 public electric vehicle charging ports, eight at the town hall parking lot "
    "and four at the grocery store on Main Street. Charging prices are lower between 10 a.m. and 3 p.m. to "
    "encourage drivers to charge when solar output is highest. A volunteer energy coach program trained 15 "
    "residents to help neighbors read their bills and understand time-of-use pricing. The high school's "
    "environmental science class used real-time data from the microgrid for a semester-long project, and two "
    "students completed summer internships with the operations team. Attendance at the annual member meeting "
    "reached a record 341 people.",
    # A7 - Challenges
    "Several challenges slowed work during the year. A firmware update to the inverters in April contained a "
    "bug that caused occasional false fault readings; the issue did not affect power delivery, but it "
    "generated hundreds of unnecessary alarms until the manufacturer released a corrected version in June. A "
    "replacement transformer for the Pell Orchard array was delivered 19 weeks later than promised because of "
    "supply chain delays, and the array operated at reduced capacity during that period. Staffing was also a "
    "constraint: the operations team has four full-time employees, and one position was vacant from May to "
    "September. Several projects, including the planned upgrade of the customer billing portal, were "
    "postponed to 2026. Finally, a proposal to add a third solar site near the river was withdrawn after a "
    "wetlands survey found that the location was home to a protected amphibian species. The board has asked "
    "staff to identify alternative sites, with a preference for rooftops and land that has already been "
    "developed, such as the parking area behind the community center.",
    # A8 - 2026 plans (needle: Marigold Storage Annex)
    "For 2026, the cooperative board approved three main initiatives. First, the microgrid will add 2 MWh of "
    "battery storage in a new building next to the inverter station, bringing total capacity to 8 MWh. In its "
    "November meeting, the board voted to name this new building the Marigold Storage Annex, after the "
    "wildflowers planted along its access road. Second, the cooperative will begin a vehicle-to-grid pilot "
    "with the school district, using four electric school buses that can return power to the microgrid "
    "during summer months when school is not in session. Third, staff will complete the delayed billing "
    "portal upgrade and add a feature that shows each member how much of their electricity came from local "
    "sources each month. The board set a target of supplying 80 percent of the town's electricity locally by "
    "the end of 2026. Staff caution that reaching the target depends on weather, on timely equipment "
    "delivery, and on how quickly winter demand grows as more households switch to heat pumps. Progress will "
    "be reported at quarterly member meetings.",
]

DOC_B_TITLE = "Ashgrove County Public Library System: 2025 Annual Report"
DOC_B = [
    # B1 - Overview
    "The Ashgrove County Public Library System operates seven branches serving about 186,000 residents. In "
    "2025, the system recorded 1.12 million in-person visits, an increase of 9 percent over 2024. Total "
    "circulation reached 2.4 million items, of which 41 percent were digital, including e-books, e-audiobooks, "
    "and streaming films. The number of active library card holders rose to 97,500. Library leadership "
    "attributes most of the growth in visits to the reopening of the Central Branch in April, after a "
    "renovation that kept the building closed for 14 months. During the closure, many regular visitors had "
    "shifted to smaller branches or to digital services, and the reopening brought a large share of them back. "
    "Digital borrowing continued to grow even after the Central Branch reopened, suggesting that many patrons "
    "now use both physical and digital collections rather than choosing one. This report covers the "
    "renovation, collections, public programs, staffing, finances, the challenges the system faced during "
    "the year, and the plans approved by the library board for 2026. All figures are for the calendar year "
    "unless otherwise noted.",
    # B2 - Central Branch renovation
    "The Central Branch renovation cost 18.6 million dollars. Most of the funding came from a bond measure "
    "approved by county voters in 2022, which passed with 61 percent of the vote; private donations "
    "contributed a further 2.1 million dollars. The renovated building includes a makerspace with three 3D "
    "printers, a laser cutter, and sewing and electronics stations, as well as 12 bookable study rooms. The "
    "children's wing was doubled in size and now includes a dedicated story-time room with acoustic panels. "
    "A new rooftop reading garden, open from April through October, quickly became one of the most "
    "photographed spots in the city. Solar panels on the south side of the roof, combined with LED lighting "
    "and an upgraded ventilation system, have reduced the building's electricity use by about 30 percent "
    "compared with the old building. Opening day on April 12 drew about 4,800 visitors, and staff issued "
    "more than 600 new library cards that weekend. The makerspace was fully booked for its first six weeks, "
    "and the library has since added evening sessions to meet demand.",
    # B3 - Collections and digital services
    "The library added about 38,000 new physical items to its collections in 2025 and removed roughly 45,000 "
    "outdated or damaged items, most of which were sold at book sales or sent for recycling. E-book loans "
    "reached 640,000, the highest annual total so far. In May, the library launched a Library of Things "
    "collection with 220 items that can be borrowed like books, including telescopes, sewing machines, "
    "board games, power tools, and portable Wi-Fi hotspots. The 150 hotspots were the most requested items "
    "in the collection, and the average wait for one was about three weeks by the end of the year. To reduce "
    "the wait, the library shortened the hotspot loan period from three weeks to two. Staff also expanded "
    "the world languages collection, adding titles in Spanish, Vietnamese, Korean, and Amharic based on "
    "patron requests and census data. A new catalog search feature now shows whether an item is on the shelf "
    "at each branch in real time, which reduced phone calls asking about availability by about a quarter.",
    # B4 - Programs
    "The library offered about 6,900 public programs in 2025, with total attendance of 142,000. The summer "
    "reading challenge enrolled a record 11,300 children, and 64 percent of them completed the challenge by "
    "reading or listening for at least 20 hours. The adult literacy program served 280 learners with the help "
    "of 96 volunteer tutors, and 41 learners passed a high school equivalency exam during the year. The job "
    "and career center held about 1,900 one-on-one sessions covering resumes, online applications, and "
    "interview practice; 214 participants later reported that they had found a job. Technology classes for "
    "older adults, covering topics such as video calls and avoiding online scams, were among the fastest "
    "growing programs, with attendance up 38 percent. Teen programs included a weekly coding club and a "
    "youth advisory board that helped plan the new makerspace schedule. Staff noted that evening and "
    "weekend programs consistently drew larger audiences than weekday daytime sessions, a pattern that is "
    "shaping the program calendar for 2026.",
    # B5 - Staffing, hours, bookmobile
    "The library system employs 164 staff members, of whom 118 work full time. Staff turnover was 8 percent, "
    "down from 13 percent in 2024, which leadership credits partly to a new pay scale adopted in January. "
    "Starting in September, four branches extended their hours to include Sunday afternoons, a change that "
    "had been one of the most common requests in patron surveys. In June, the library took delivery of a new "
    "electric bookmobile, which replaced a diesel vehicle that had been in service for 19 years. The "
    "bookmobile now makes 23 scheduled stops each week, including senior centers, rural schools, and two "
    "apartment complexes that are far from any branch. Because the new vehicle has a wheelchair lift and "
    "onboard Wi-Fi, it can also host small computer help sessions. Volunteers contributed about 21,000 hours "
    "of service during the year, mostly shelving, tutoring, and helping run the book sales. Staff training "
    "focused on customer service, digital privacy, and supporting patrons in crisis, in partnership with a "
    "local social services agency.",
    # B6 - Finances and fine-free policy
    "The library's operating budget for 2025 was 21.4 million dollars. About 78 percent came from the county, "
    "9 percent from the state, and the rest from grants, donations, and other sources; competitive grants "
    "totaled 1.3 million dollars. In January 2025, the library eliminated overdue fines on all materials. "
    "Fines had previously brought in about 310,000 dollars per year, but staff found that the cost of "
    "collecting them and the number of patrons who stopped using the library because of unpaid fines "
    "outweighed the revenue. After the change, 4,100 previously blocked accounts were reactivated, and the "
    "share of items returned within a month of their due date rose to 96 percent. Patrons are still charged "
    "for items that are lost or damaged. Personnel costs made up about 64 percent of spending, collections "
    "about 14 percent, and facilities and technology most of the remainder. The board approved a modest "
    "increase in the collections budget for 2026, with most of the new funding going to digital materials "
    "and the Library of Things.",
    # B7 - Challenges
    "The year also brought several setbacks. In February, a roof leak at the Riverside Branch damaged part "
    "of the collection and closed the branch for five weeks while repairs were made. In August, the heating "
    "and cooling system at the Northgate Branch failed during a heat wave, and the branch closed early on "
    "six days until a replacement unit was installed. In October, the library was targeted by a phishing "
    "attack in which a staff member's email account was compromised. The IT team contained the incident "
    "within six hours, and an outside review found that no patron data had been exposed; however, the "
    "online catalog was taken offline for two days as a precaution. The library has since required "
    "two-factor authentication for all staff accounts. Parking remains a frequent complaint at the Central "
    "Branch, especially on Saturdays, and the library is negotiating with the county to share a nearby "
    "government parking garage on weekends. Finally, rising costs for e-book licenses continue to strain the "
    "digital budget.",
    # B8 - 2026 plans (needle: Lantern Room)
    "For 2026, the library board approved plans to open an eighth branch, the Willow Creek Branch, in a "
    "former bank building on Orchard Avenue. Renovation is scheduled to finish in October 2026. The branch's "
    "children's reading room will be called the Lantern Room, a name chosen in a vote by about 3,200 local "
    "students. The library also plans to expand the Library of Things to about 400 items, add more hotspots "
    "to shorten the waiting list, and launch an online homework help service staffed by tutors until 9 p.m. "
    "on school nights. The board set a goal of 1.2 million in-person visits for 2026. Staff caution that "
    "meeting the goal depends on the Willow Creek Branch opening on schedule and on resolving the parking "
    "problem at the Central Branch. A community survey will be conducted in the spring to guide decisions "
    "about branch hours and new services, and the results will be presented at the board's June meeting.",
]

DOCS = {"A": (DOC_A_TITLE, DOC_A), "B": (DOC_B_TITLE, DOC_B)}
ALL = range(1, 9)


def passage(doc, blocks):
    title, body = DOCS[doc]
    return title + "\n\n" + "\n\n".join(body[b - 1] for b in blocks)


def ctx(doc, blocks, question):
    return ("Read the report below and answer the question that follows.\n\n"
            f"{passage(doc, blocks)}\n\nQuestion: {question}\nAnswer:")


# ======================================================================================
# PERFORMANCE SET (50) - greedy, ignore_eos, fixed n_predict, cache_prompt off.
# "core": True marks the 12 prompts used for the full warm-up + 3 measured-run protocol.
# ======================================================================================
FS_SENTIMENT = (
    "Classify the sentiment of each review as positive, negative, or neutral.\n\n"
    "Review: The battery lasts all day and the screen is gorgeous.\nSentiment: positive\n\n"
    "Review: It stopped charging after two weeks and support never replied.\nSentiment: negative\n\n"
    "Review: The package arrived on Tuesday.\nSentiment: neutral\n\n"
    "Review: Setup took five minutes and it just works.\nSentiment: positive\n\n"
    "Review: The keyboard feels cheap and two keys already stick.\nSentiment: negative\n\n"
    "Review: It comes in black and silver.\nSentiment: neutral\n\n"
    "Review: Honestly the best headphones I have owned for the price.\nSentiment: positive\n\n"
    "Review: The app logs me out every time I close it, which is infuriating.\nSentiment: negative\n\n"
    "Review: Sound quality is decent, but the case scratches far too easily and the hinge creaks.\n"
    "Sentiment:")
FS_EXTRACT = (
    "Extract the person, date, and amount from each sentence as JSON.\n\n"
    "Sentence: Maria Lopez paid $45.00 on March 3, 2024.\n"
    'JSON: {"person": "Maria Lopez", "date": "2024-03-03", "amount": 45.00}\n\n'
    "Sentence: On July 19, 2023, a refund of $120.50 was issued to Tom Becker.\n"
    'JSON: {"person": "Tom Becker", "date": "2023-07-19", "amount": 120.50}\n\n'
    "Sentence: Priya Nair transferred $9.99 for the app subscription on January 2, 2025.\n"
    'JSON: {"person": "Priya Nair", "date": "2025-01-02", "amount": 9.99}\n\n'
    "Sentence: The invoice for $310.00 was settled by Jonah Kim on November 28, 2024.\n"
    'JSON: {"person": "Jonah Kim", "date": "2024-11-28", "amount": 310.00}\n\n'
    "Sentence: On September 5, 2025, Aisha Bello sent $72.25 to cover the team dinner.\nJSON:")
FS_TOPIC = (
    "Classify each headline into one topic: sports, science, business, or entertainment.\n\n"
    "Headline: Local team clinches the championship in overtime thriller\nTopic: sports\n\n"
    "Headline: Researchers map a new species of deep-sea coral\nTopic: science\n\n"
    "Headline: Regional bakery chain opens its fiftieth store\nTopic: business\n\n"
    "Headline: Animated film breaks opening-weekend records\nTopic: entertainment\n\n"
    "Headline: Marathon runner sets a new course record despite heavy rain\nTopic: sports\n\n"
    "Headline: Telescope captures a rare double-star eclipse\nTopic: science\n\n"
    "Headline: Shipping company reports higher quarterly profits\nTopic: business\n\n"
    "Headline: Indie band announces a surprise reunion tour\nTopic: entertainment\n\n"
    "Headline: Startup raises funding to build cheaper battery storage for homes\nTopic:")
FS_TRANSLATE = (
    "Translate English to French.\n\n"
    "English: Good morning, how are you?\nFrench: Bonjour, comment allez-vous ?\n\n"
    "English: The library opens at nine o'clock.\nFrench: La bibliothèque ouvre à neuf heures.\n\n"
    "English: I would like a cup of coffee, please.\nFrench: Je voudrais une tasse de café, s'il vous plaît.\n\n"
    "English: We are going to the beach on Saturday.\nFrench: Nous allons à la plage samedi.\n\n"
    "English: My sister lives in a small village near the mountains.\nFrench: Ma sœur habite dans un petit "
    "village près des montagnes.\n\n"
    "English: The train was late because of the snow.\nFrench:")
FS_CITIES = (
    "List every city mentioned in each sentence, separated by commas.\n\n"
    "Sentence: She flew from Chicago to Denver and then drove to Boulder.\nCities: Chicago, Denver, Boulder\n\n"
    "Sentence: The conference moved from Lisbon to Madrid this year.\nCities: Lisbon, Madrid\n\n"
    "Sentence: Our team has offices in Toronto, Seattle, and Austin.\nCities: Toronto, Seattle, Austin\n\n"
    "Sentence: After a week in Kyoto, they took the train to Osaka.\nCities: Kyoto, Osaka\n\n"
    "Sentence: He grew up in Nairobi, studied in Cairo, and now works between Dubai and Mumbai.\nCities:")
FS_REWRITE = (
    "Rewrite each formal sentence in a casual, friendly tone.\n\n"
    "Formal: We regret to inform you that the event has been postponed.\n"
    "Casual: Heads up, the event got pushed back.\n\n"
    "Formal: Kindly submit your report by the end of the business day.\n"
    "Casual: Can you send your report over by end of day?\n\n"
    "Formal: Your request has been received and is currently under review.\n"
    "Casual: Got your request, we're looking at it now.\n\n"
    "Formal: Please be advised that the office will be closed on Monday.\n"
    "Casual: Just so you know, the office is closed Monday.\n\n"
    "Formal: We would appreciate your attendance at the quarterly planning meeting.\nCasual:")

SWEEP = {  # same question at 256 / 1024 / 2048 tokens; answers are all in block 1
    "A": ["Why did the share of electricity supplied by the microgrid increase in 2025, and by how much?",
          "How many households and businesses does the microgrid serve?",
          "What generation and storage equipment does the microgrid use?"],
    "B": ["Why did in-person library visits increase in 2025?",
          "What share of total circulation was digital in 2025?",
          "How many branches does the library system have, and how many residents does it serve?"],
}
SWEEP_BLOCKS = {"ctx256": [1], "ctx1024": [1, 2, 3, 4], "ctx2048": list(ALL)}


def build_perf():
    rows = []

    def add(cat, bucket, n, user, core=False):
        rows.append(dict(category=cat, bucket=bucket, n_predict=n, user=user, core=core))

    qa = ["What is the difference between a CPU and a GPU? Answer in two or three sentences.",
          "Explain what a hash table is and give one example of where it is used in real software.",
          "What causes the seasons on Earth? Answer in two or three sentences.",
          "What is the difference between RAM and storage in a computer?",
          "Explain what an API is to someone who is new to programming.",
          "How does a vaccine train the immune system? Answer briefly.",
          "What is inflation, and why does it matter to households?",
          "How is machine learning different from traditional rule-based programming?"]
    for i, u in enumerate(qa):
        add("qa", "short", 128, u, core=i < 2)

    reasoning = [
        "A train leaves a station at 2:15 p.m. and travels 180 km at a constant 72 km/h. At what time does it "
        "arrive? Show your steps.",
        "Four friends, Ana, Ben, Chloe, and Dev, each own exactly one different pet: a cat, a dog, a parrot, or "
        "a fish. Ana is allergic to fur. Ben's pet can talk. Chloe walks her pet every morning. Who owns which "
        "pet? Explain your reasoning step by step.",
        "A store discounts an $80 jacket by 25 percent and then adds 8 percent sales tax. What is the final "
        "price? Show your steps.",
        "If today is Wednesday, what day of the week will it be 100 days from now? Explain your reasoning.",
        "Pipe A fills a tank in 6 hours and pipe B fills it in 4 hours. How long does it take with both pipes "
        "open? Show your steps.",
        "Three boxes are labeled 'apples', 'oranges', and 'mixed', and every label is wrong. You may take one "
        "fruit from one box without looking inside. How can you relabel all three boxes correctly? Explain."]
    for i, u in enumerate(reasoning):
        add("reasoning", "short", 256, u, core=i < 2)

    for i, u in enumerate([FS_SENTIMENT, FS_EXTRACT, FS_TOPIC, FS_TRANSLATE, FS_CITIES, FS_REWRITE]):
        add("few_shot", "medium", 64, u, core=i < 2)

    code = ["Write a Python function that returns the n-th Fibonacci number using iteration, with a docstring "
            "and input validation.",
            "Write a Python function that merges two sorted lists into one sorted list without using sort() "
            "or sorted().",
            "Write a SQL query that returns the top 5 customers by total order amount, given tables "
            "customers(id, name) and orders(id, customer_id, amount).",
            "Write a JavaScript function debounce(fn, delay) and briefly explain how it works.",
            "Write a Python class Stack with push, pop, peek, and is_empty methods, including docstrings."]
    for i, u in enumerate(code):
        add("code", "short", 256, u, core=i == 0)

    gen = ["Write a 300-word explanation of how photosynthesis works for a high school student.",
           "Write a 300-word short story about a lighthouse keeper who finds a message in a bottle.",
           "Write a 300-word explanation of how a message travels across the internet from one computer to "
           "another.",
           "Write a 300-word guide to preparing for a job interview.",
           "Write a 300-word explanation of how a refrigerator keeps food cold."]
    for i, u in enumerate(gen):
        add("generation", "short", 512, u, core=i == 0)

    for doc, questions in SWEEP.items():
        for qi, q in enumerate(questions):
            for bucket, blocks in SWEEP_BLOCKS.items():
                add("long_context", bucket, 128, ctx(doc, blocks, q), core=(doc == "A" and qi == 0))

    for doc in ("A", "B"):
        add("long_context", "ctx2048", 256,
            "Summarize the report below in five bullet points.\n\n" + passage(doc, ALL), core=doc == "A")

    for i, r in enumerate(rows, 1):
        r["id"] = f"P{i:02d}"
    return rows


# ======================================================================================
# QUALITY SET (100) - held out, greedy, stop at EOS. IDs are prefixed by category:
#   M math(20) L logic(8) F factual(15) R reading(15) X long-context(12)
#   I instruction(12) S few-shot(6) C code(6) O open-ended/manual(6)
# ======================================================================================
ANS = " End with a final line in the form 'Answer: <number>'."
YN = " Answer 'yes' or 'no' first, then give one sentence of explanation."


def num(v, tol=0.01):
    return {"type": "number", "value": v, "tol": tol}


def has(*vals, cs=False):
    return {"type": "contains_all", "values": list(vals), "case_sensitive": cs}


def anyof(*vals):
    return {"type": "contains_any", "values": list(vals)}


def rx(*pats):
    return {"type": "regex_all", "patterns": list(pats)}


def tests(*ts):
    return {"type": "python_tests", "tests": list(ts)}


def manual(rubric):
    return {"type": "manual", "rubric": rubric}


FS_TICKETS = (
    "Classify each support ticket as billing, technical, or account. Reply with the category only.\n\n"
    "Ticket: I was charged twice for my subscription this month.\nCategory: billing\n\n"
    "Ticket: The app crashes whenever I open the settings page.\nCategory: technical\n\n"
    "Ticket: How do I change the email address on my profile?\nCategory: account\n\n"
    "Ticket: My invoice shows the wrong company name.\nCategory: billing\n\n"
    "Ticket: I forgot my password and the reset link isn't arriving.\nCategory: account\n\n"
    "Ticket: Videos won't load on my phone but work on my laptop.\nCategory: technical\n\n")
FS_ANIMALS = (
    "Classify each animal as mammal, bird, reptile, or fish. Reply with the class only.\n\n"
    "Animal: dolphin\nClass: mammal\n\nAnimal: eagle\nClass: bird\n\nAnimal: iguana\nClass: reptile\n\n"
    "Animal: salmon\nClass: fish\n\nAnimal: bat\nClass: mammal\n\nAnimal: crocodile\nClass: reptile\n\n"
    "Animal: penguin\nClass:")
FS_ANTONYM = ("Give the opposite of each word. Reply with one word only.\n\n"
              "Word: hot\nOpposite: cold\n\nWord: tall\nOpposite: short\n\nWord: happy\nOpposite: sad\n\n"
              "Word: open\nOpposite: closed\n\nWord: early\nOpposite:")
FS_CAPITAL = ("Give the capital city of each country. Reply with the city only.\n\n"
              "Country: France\nCapital: Paris\n\nCountry: Japan\nCapital: Tokyo\n\n"
              "Country: Kenya\nCapital: Nairobi\n\nCountry: Peru\nCapital: Lima\n\nCountry: Canada\nCapital:")
FS_PLURAL = ("Give the plural of each noun. Reply with one word only.\n\n"
             "Noun: child\nPlural: children\n\nNoun: mouse\nPlural: mice\n\nNoun: goose\nPlural: geese\n\n"
             "Noun: tooth\nPlural: teeth\n\nNoun: foot\nPlural:")


def label(v):
    return {"type": "label", "value": v}


QUALITY_SPEC = {
    "M": ("math", 256, [
        ("A bakery sells muffins for $3.00 each and cookies for $1.25 each. Maya buys 4 muffins and 6 cookies "
         "and pays with a $20 bill. How much change does she get?" + ANS, num(0.5)),
        ("A rectangle has a perimeter of 46 cm and a length of 15 cm. What is its area in square centimeters?"
         + ANS, num(120)),
        ("If 3 workers can paint 3 fences in 3 hours, how many hours do 6 workers need to paint 6 fences, "
         "working at the same rate?" + ANS, num(3)),
        ("What is 17% of 340?" + ANS, num(57.8)),
        ("What is the next number in the sequence 2, 6, 12, 20, 30, ...?" + ANS, num(42)),
        ("A car uses 6 liters of fuel per 100 km. How many liters does it use to drive 350 km?" + ANS, num(21)),
        ("Sam is 4 times as old as his son. In 20 years, Sam will be twice as old as his son. How old is Sam "
         "now?" + ANS, num(40)),
        ("The average of five numbers is 18. Four of the numbers are 12, 20, 15, and 25. What is the fifth "
         "number?" + ANS, num(18)),
        ("A shirt costs $24 after a 20% discount. What was its original price in dollars?" + ANS, num(30)),
        ("How many minutes are there in 3.5 hours?" + ANS, num(210)),
        ("A recipe uses 3 eggs for every 12 cookies. How many eggs are needed for 60 cookies?" + ANS, num(15)),
        ("Two trains start 300 km apart and travel toward each other, one at 60 km/h and the other at 90 km/h. "
         "After how many hours do they meet?" + ANS, num(2)),
        ("What is the sum of all whole numbers from 1 to 50?" + ANS, num(1275)),
        ("A rectangle is 8 cm long and 6 cm wide. How long is its diagonal in centimeters?" + ANS, num(10)),
        ("How much simple interest, in dollars, does $1,000 earn in 3 years at 5% per year?" + ANS, num(150)),
        ("A class has 30 students, and 40% of them are boys. How many girls are in the class?" + ANS, num(18)),
        ("If 2x + 7 = 31, what is x?" + ANS, num(12)),
        ("A rectangular garden is 12 m by 9 m. A fence around its edge has a post every 3 m, including one at "
         "each corner. How many posts are there?" + ANS, num(14)),
        ("Lena reads 25 pages per day. Her book has 340 pages. How many days does she need to finish it?" + ANS,
         num(14)),
        ("What is 3/8 written as a percentage?" + ANS, num(37.5)),
    ]),
    "L": ("logic", 128, [
        ("All bloops are razzies. Some razzies are lazzies. Can we conclude that some bloops are lazzies?" + YN,
         {"type": "yesno", "value": "no"}),
        ("If it rains, the ground gets wet. The ground is wet. Can we conclude that it rained?" + YN,
         {"type": "yesno", "value": "no"}),
        ("All roses are flowers. All flowers need water. Do roses need water?" + YN,
         {"type": "yesno", "value": "yes"}),
        ("Tom is taller than Jim. Jim is taller than Ann. Is Ann taller than Tom?" + YN,
         {"type": "yesno", "value": "no"}),
        ("No fish are mammals. All whales are mammals. Can a whale be a fish?" + YN,
         {"type": "yesno", "value": "no"}),
        ("Every member of the chess club also plays piano. Maria is in the chess club. Does Maria play piano?"
         + YN, {"type": "yesno", "value": "yes"}),
        ("Five runners finished a race. D finished first and E finished last. A finished before B, and C "
         "finished after B. Who finished third? End with a final line in the form 'Answer: <letter>'.",
         {"type": "answer_text", "value": "B"}),
        ("A farmer has 17 sheep. All but 9 run away. How many sheep are left?" + ANS, num(9)),
    ]),
    "F": ("factual", 64, [
        ("What is the chemical symbol for sodium?", has("Na", cs=True)),
        ("Which planet in our solar system has the shortest year?", has("Mercury")),
        ("In what year did the Berlin Wall fall?", has("1989")),
        ("Which gas do plants take in from the air for photosynthesis?", anyof("carbon dioxide", "CO2")),
        ("Who wrote the novel 'Pride and Prejudice'?", has("Austen")),
        ("What is the capital of Australia?", has("Canberra")),
        ("How many bones are in the adult human body?", has("206")),
        ("What is the largest ocean on Earth?", has("Pacific")),
        ("At sea level, what is the boiling point of water in degrees Celsius?", has("100")),
        ("Which element has atomic number 1?", has("hydrogen")),
        ("Who developed the theory of general relativity?", has("Einstein")),
        ("What is the square root of 144?", has("12")),
        ("On which continent is the Sahara Desert?", has("Africa")),
        ("What is the hardest naturally occurring substance?", has("diamond")),
        ("How many continents are there on Earth by the most common convention?", anyof("7", "seven")),
    ]),
    "R": ("reading", 128, [
        (ctx("A", [4], "How long did the August outage last, and what caused it?"),
         rx(r"3\s*(hours?|hrs?)\D{0,12}40\s*min", r"lightning")),
        (ctx("A", [1, 2], "What share of the town's electricity did the microgrid supply in 2024?"),
         rx(r"\b58\s*(%|percent)")),
        (ctx("A", [2], "In which month did solar output peak, and how much was produced?"), rx(r"June", r"812")),
        (ctx("A", [3], "How many full equivalent charge cycles did the battery system complete in 2025?"),
         has("287")),
        (ctx("A", [3], "What did the battery management system do when the cooling fan failed?"),
         anyof("half", "halved", "50")),
        (ctx("A", [5], "How did Kestrel Valley's average residential rate compare with the regional average?"),
         rx(r"13\.8", r"16\.1")),
        (ctx("A", [6], "How many households used the heat pump rebate program in 2025?"), has("214")),
        (ctx("A", [7], "How late was the replacement transformer for the Pell Orchard array?"),
         rx(r"\b19\s*weeks")),
        (ctx("B", [1], "How many active library card holders were there in 2025?"), rx(r"97,?500")),
        (ctx("B", [2], "How much did the Central Branch renovation cost, and where did most of the money come "
                       "from?"), rx(r"18\.6", r"bond")),
        (ctx("B", [3], "How many items were in the Library of Things when it launched?"), has("220")),
        (ctx("B", [4], "How many children enrolled in the summer reading challenge, and what share completed "
                       "it?"), rx(r"11,?300", r"\b64\s*(%|percent)")),
        (ctx("B", [5], "What kind of vehicle replaced the old diesel bookmobile?"), has("electric")),
        (ctx("B", [6], "In which month did the library eliminate overdue fines?"), has("January")),
        (ctx("B", [7], "How long was the Riverside Branch closed after the roof leak?"),
         rx(r"\b(5|five)\s*weeks")),
    ]),
    "X": ("long_context", 192, [
        (ctx("A", ALL, "What name did the board give to the new battery building?"), has("Marigold")),
        (ctx("A", ALL, "How much total battery storage did the microgrid have before March 2025, and how much "
                       "will it have after the 2026 expansion?"),
         rx(r"\b3\s*(MWh|megawatt)", r"\b8\s*(MWh|megawatt)")),
        (ctx("A", ALL, "How many electric school buses will take part in the vehicle-to-grid pilot?"),
         anyof("4", "four")),
        (ctx("A", ALL, "Why was the proposed third solar site near the river withdrawn?"), has("amphibian")),
        (ctx("A", ALL, "How many people attended the annual member meeting?"), has("341")),
        (ctx("A", ALL, "How many unplanned outages occurred in 2025, and what caused the longest one?"),
         rx(r"\b(3|three)\b", r"lightning")),
        (ctx("B", ALL, "What will the children's reading room at the new branch be called?"), has("Lantern")),
        (ctx("B", ALL, "How many staff does the library system employ, and how many work full time?"),
         rx(r"\b164\b", r"\b118\b")),
        (ctx("B", ALL, "How quickly did the IT team contain the phishing incident?"),
         rx(r"\b(6|six)\s*hours")),
        (ctx("B", ALL, "How many branches does the system have now, and how many will it have after the "
                       "2026 opening?"), rx(r"\b(7|seven)\b", r"\b(8|eight)")),
        (ctx("B", ALL, "What percentage of the vote did the 2022 bond measure receive?"),
         rx(r"\b61\s*(%|percent)")),
        (ctx("B", ALL, "How many previously blocked accounts were reactivated after fines were eliminated?"),
         rx(r"4,?100")),
    ]),
    "I": ("instruction", 128, [
        ("List exactly three benefits of regular exercise as a bulleted list. Start each line with '- ' and "
         "write nothing else.", {"type": "bullets_exact", "n": 3}),
        ("Return a JSON object with exactly these keys: \"city\", \"country\", \"population_millions\", "
         "describing Tokyo. Output only the JSON.",
         {"type": "json_keys", "keys": ["city", "country", "population_millions"]}),
        ("Describe the ocean in one sentence of at most 20 words.", {"type": "one_sentence_max_words", "n": 20}),
        ("List exactly five fruits as a numbered list (1., 2., ...). Write nothing else.",
         {"type": "numbered_exact", "n": 5}),
        ("Write the word banana in all capital letters and nothing else.", {"type": "exact", "value": "BANANA"}),
        ("How many letters are in the word 'computer'? Reply with the number only.",
         {"type": "exact", "value": "8"}),
        ("Write one sentence about cats that begins with the word 'Yesterday'.",
         {"type": "starts_with", "value": "Yesterday"}),
        ("Write a short poem about rain that is exactly three lines long, with no title.",
         {"type": "line_count", "n": 3}),
        ("Return a JSON array of the three primary colors of light as lowercase strings. Output only the JSON.",
         {"type": "json_set", "values": ["red", "green", "blue"]}),
        ("Using only lowercase letters, tell me the color of the sky on a clear day.",
         {"type": "all_lower", "values": ["blue"]}),
        ("Give two reasons to learn a second language. Put each on its own line starting with 'Reason:' and "
         "write nothing else.", {"type": "prefixed_lines", "prefix": "Reason:", "n": 2}),
        ("Write a product description for a reusable water bottle in exactly three sentences.",
         {"type": "sentence_count", "n": 3}),
    ]),
    "S": ("few_shot", 16, [
        (FS_TICKETS + "Ticket: After the latest update, the export button does nothing when I click it.\n"
                      "Category:", label("technical")),
        (FS_TICKETS + "Ticket: I'd like a refund for the annual plan I bought yesterday.\nCategory:",
         label("billing")),
        (FS_ANIMALS, label("bird")),
        (FS_ANTONYM, label("late")),
        (FS_CAPITAL, label("ottawa")),
        (FS_PLURAL, label("feet")),
    ]),
    "C": ("code", 256, [
        ("Write a Python function is_palindrome(s) that returns True if the string s reads the same forwards "
         "and backwards, ignoring case and any non-alphanumeric characters. Output only the code.",
         tests("assert is_palindrome('A man, a plan, a canal: Panama') is True",
               "assert is_palindrome('race a car') is False", "assert is_palindrome('') is True",
               "assert is_palindrome('No lemon, no melon') is True", "assert is_palindrome('ab') is False")),
        ("Write a Python function fizzbuzz(n) that returns a list of strings for the numbers 1 to n: 'Fizz' for "
         "multiples of 3, 'Buzz' for multiples of 5, 'FizzBuzz' for multiples of both, otherwise the number as "
         "a string. Output only the code.",
         tests("assert fizzbuzz(5) == ['1', '2', 'Fizz', '4', 'Buzz']", "assert fizzbuzz(15)[-1] == 'FizzBuzz'",
               "assert len(fizzbuzz(30)) == 30", "assert fizzbuzz(0) == []")),
        ("Write a Python function count_vowels(s) that returns the number of vowels (a, e, i, o, u, either "
         "case) in s. Output only the code.",
         tests("assert count_vowels('hello') == 2", "assert count_vowels('AEIOU') == 5",
               "assert count_vowels('rhythm') == 0", "assert count_vowels('') == 0")),
        ("Write a Python function second_largest(nums) that returns the second largest distinct value in a "
         "list, or None if there is no such value. Output only the code.",
         tests("assert second_largest([3, 1, 4, 4, 2]) == 3", "assert second_largest([5, 5]) is None",
               "assert second_largest([7]) is None", "assert second_largest([-1, -2, -3]) == -2")),
        ("Write a Python function reverse_words(s) that returns the words of s in reverse order, separated by "
         "single spaces. Output only the code.",
         tests("assert reverse_words('hello world') == 'world hello'",
               "assert reverse_words('one two three') == 'three two one'", "assert reverse_words('solo') == 'solo'",
               "assert reverse_words('  spaced   out  ') == 'out spaced'")),
        ("Write a Python function is_prime(n) that returns True if n is a prime number and False otherwise. "
         "Output only the code.",
         tests("assert is_prime(2) is True", "assert is_prime(17) is True", "assert is_prime(1) is False",
               "assert is_prime(0) is False", "assert is_prime(21) is False", "assert is_prime(97) is True")),
    ]),
    "O": ("open_ended", 192, [
        ("Explain to a 10-year-old why the sky is blue, in three or four sentences.",
         manual("Sunlight has all colors; air scatters short blue wavelengths more than red; scattered blue "
                "light reaches us from all directions. Correct, 3-4 sentences, kid-friendly.")),
        ("Summarize the following section of a report in two sentences.\n\n" + DOC_A[2],
         manual("287 cycles; 2.1% loss in original bank, none in new; July fan failure halved discharge, no "
                "damage; response: remote alerts, spare fans, filter checks. Two sentences, no invented facts.")),
        ("Explain the difference between weather and climate in three sentences.",
         manual("Weather = short-term conditions (hours/days); climate = long-term average patterns (decades) "
                "for a region; a correct example or contrast. Exactly three sentences.")),
        ("Give a first-year college student advice on managing their time, in four sentences.",
         manual("Concrete, practical tips (planning/calendar, prioritizing, breaking up big tasks, sleep or "
                "breaks); coherent, not generic filler. Four sentences.")),
        ("Summarize the following section of a report in two sentences.\n\n" + DOC_B[1],
         manual("$18.6M cost, mostly 2022 bond (61%) + $2.1M donations; new makerspace, study rooms, larger "
                "children's wing, rooftop garden; ~30% less electricity; April 12 opening drew ~4,800. Two "
                "sentences, no invented facts.")),
        ("In three or four sentences, explain how a moving bicycle stays upright, for a general audience.",
         manual("Mentions steering corrections by the rider and/or the bike's self-stabilizing geometry "
                "(trail, gyroscopic effects play a minor role); no major physics errors; 3-4 sentences.")),
    ]),
}

RUBRIC_SCALE = {
    5: "Fully correct and complete, follows all instructions, fluent.",
    4: "Correct with a minor omission or slight awkwardness.",
    3: "Partly correct: one key point missing or one small error.",
    2: "Mostly wrong or ignores an instruction, but on topic.",
    1: "Wrong, off topic, repetitive/garbled, or hallucinated facts.",
}


def build_quality():
    rows = []
    for prefix, (cat, n_predict, items) in QUALITY_SPEC.items():
        for i, (user, check) in enumerate(items, 1):
            rows.append(dict(id=f"{prefix}{i:02d}", category=cat, n_predict=n_predict, user=user, check=check))
    return rows


def count_tokens(text, model, tokenizer_bin, tmp):
    if model and tokenizer_bin:
        tmp.write_text(text, encoding="utf-8")
        out = subprocess.run([tokenizer_bin, "-m", model, "-f", str(tmp), "--ids", "--log-disable"],
                             capture_output=True, text=True)
        lists = re.findall(r"\[([\d,\s]*)\]", out.stdout)
        if lists:
            return len(re.findall(r"\d+", lists[-1])), "exact"
        print(f"warning: could not parse llama-tokenize output; using estimate\n{out.stderr[-300:]}",
              file=sys.stderr)
    return round(len(text) / 4), "estimate"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="GGUF file used for exact token counts")
    ap.add_argument("--tokenizer-bin", default=shutil.which("llama-tokenize"),
                    help="path to llama.cpp's llama-tokenize binary")
    args = ap.parse_args()

    perf, quality = build_perf(), build_quality()

    (ROOT / "perf").mkdir(exist_ok=True)
    for old in (ROOT / "perf").glob("*.txt"):
        old.unlink()
    (ROOT / "passage").mkdir(exist_ok=True)
    for old in (ROOT / "passage").glob("*.txt"):
        old.unlink()
    for doc in DOCS:
        (ROOT / "passage" / f"report_{doc.lower()}.txt").write_text(passage(doc, ALL) + "\n", encoding="utf-8")

    with open(ROOT / "perf_prompts.jsonl", "w", encoding="utf-8") as f:
        for p in perf:
            rec = {"id": p["id"], "core": p["core"], "category": p["category"], "prefill_bucket": p["bucket"],
                   "n_predict": p["n_predict"], "ignore_eos": True, "cache_prompt": False,
                   "temperature": 0.0, "seed": 42, "system": SYSTEM, "user": p["user"],
                   "file": f"perf/{p['id']}.txt"}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            (ROOT / "perf" / f"{p['id']}.txt").write_text(p["user"], encoding="utf-8")

    with open(ROOT / "quality_prompts.jsonl", "w", encoding="utf-8") as f:
        for q in quality:
            rec = {"id": q["id"], "category": q["category"], "n_predict": q["n_predict"], "ignore_eos": False,
                   "cache_prompt": False, "temperature": 0.0, "seed": 42, "system": SYSTEM, "user": q["user"],
                   "check": q["check"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    (ROOT / "rubric.json").write_text(json.dumps(RUBRIC_SCALE, indent=2), encoding="utf-8")

    tmp = ROOT / ".tok_tmp.txt"
    tag = Path(args.model).stem if args.model else "estimate"
    for old in ROOT.glob("token_counts_estimate.csv"):
        old.unlink()
    out_csv = ROOT / f"token_counts_{tag}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["set", "id", "core", "bucket_or_category", "user_tokens", "method", "n_predict"])
        for p in perf:
            n, how = count_tokens(p["user"], args.model, args.tokenizer_bin, tmp)
            w.writerow(["perf", p["id"], p["core"], p["bucket"], n, how, p["n_predict"]])
        for q in quality:
            n, how = count_tokens(q["user"], args.model, args.tokenizer_bin, tmp)
            w.writerow(["quality", q["id"], "", q["category"], n, how, q["n_predict"]])
    tmp.unlink(missing_ok=True)

    print(f"perf: {len(perf)} prompts ({sum(p['core'] for p in perf)} core) "
          f"{dict(Counter(p['category'] for p in perf))}")
    print(f"quality: {len(quality)} prompts {dict(Counter(q['category'] for q in quality))}")
    print(f"total: {len(perf) + len(quality)}  ->  {out_csv.name}")


if __name__ == "__main__":
    main()
