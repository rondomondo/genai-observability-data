Hey Guilherme. Unrelated to the current things we were discussing I have something here that might be of interest to Andela and to run by you to get your thoughts. 

It's email related again...

Once a person joins the Andela community etc., there are various email outreaches that we get. Take for example the latest email I got, it is to alert us about useful info and resources to be aware of, FAQs and the like.

In particular the latest email I noticed the Andela automated community outreach is pulling `skills reports` that are heavily outdated for the current 2026 AI/Tech landscape. Content is 2024 ish. My immediate instinct is to mention it to you and suggest a way to solve this for Andela. 


In a nutshell my well worn go to solution would be to build an automated QA pipeline using a Python gmail watcher, RAG, and LLM Evals to catch this stale content before it hits our talent inboxes for real, while simultaneously it can be tuning the Andela internal KB rankers with a strict time-decay penalty to try to avoid this going forward.


Have a read of the following way I would propose a way to address this without needing to lean on internal resources too much.

---

## 1. Automated CI/CD Content Tester (Python/RAG/Eval Idea)

This is a nice and simple because it treats **marketing content exactly like software code**. These days (in modern engineering anyway), we don't push code without running unit tests; so there is no reason why shouldn't be able to push push automated email campaigns referencing technical content without running **content unit tests**.

How would this look in practice?

* **The "Staging" Recipient:** Andela route these automated Braze/SparkPost campaigns to an internal test inbox (e.g., `campaign-qa@andela.com`).
* **The Python Gmail Watcher:** This is just a script uses the Gmail API or IMAP to pull new incoming message payloads. Tried and tested for years and near instantanious.

**The RAG & Eval Check:**
* The script extracts the text from email, content and any linked URLs (like the "in demand skills report" I got).
* It passes this text to an LLM evaluator paired with a pretty simple **RAG system connected to the *current* Andela internal market data** (actually augment this with a web-search tool to pull the latest tech stack trends of now (mid-2026)...).
* **The Evals:** You can set up specific assertion criteria like. For example:
* `assert_temporal_relevance`: So does this email content reference outdated skills reporting, deprecated frameworks, discounted best practices? (e.g., talking about early LLM prompt-engineering tricks when the world has moved entirely to agentic workflows and much more advanced reasoning models).
* `assert_recency_score`: a simpller one - is the underlying data source older than say 90 days?


* **The Result:** If the Eval fails, the campaign we should automatically paused in Braze, have an alert is sent to the content/product team, and then they can review,update and resend. This is very low impact and commitment up front onthe part of the team.

---

## 2. Overhauling Knowledge Bases and Rankers

More involved: Understanding of course that I have zero insight into the current internal architecture, I do suggest that:

The more meaty root cause of the issue is addressing outdated KB data. With the way things are moving so fast now with the AI developments stale content occurs so much faster. Back in the day content stagnation usually happens because someone wrote a great article in 2024, it got a lot of clicks and the recommendation algorithm gods anoint it so we are now trapped in a **feedback loop** - ranking it highly *because* it has historical traffic, completely ignoring that the shelf life of AI information these days is measured in months, not years.

Anyway this is luckily a well understood issue with respect to RAG so to fix this just means adjusting your ranking algorithms if in use:

* **Time-Decay Penalty:** Consider the need to introduce a aggressive time-decay factor into the search and recommendation vector equations (if and only if RAG is involved at all yet here!). A 95% relevance match from 2024 should easily lose to an 80% relevance match from last month.
* **Knowledge Graphs for Tech Stacks:** For the internal skills taxonomy and how it is represented database wise - the help center? - can it like graph structure that explicitly links dependencies and lifecycles? So if the system detects an article heavily indexing on some outdated tech, model or framework, then the knowledge graph should automatically flag it as "legacy" and suppress it from those automated outbound emails - or at least have flagged it for review.

---

This was longer than expected but sometimes it can be worth mentioning things, along with some solutions, that do not rely on much change having to be imposed on peoples time to address.

Best regards.
Dave.

