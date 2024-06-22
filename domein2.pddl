(define (domain sokoban)
  (:requirements :strips :typing)
  (:types location)

  (:predicates
    (at-robot ?loc - location)
    (at-box ?box - location)
    (goal ?loc - location)
    (adjacent ?from ?to - location)
    (wall ?loc - location)
  )

  (:action move
    :parameters (?from ?to - location)
    :precondition (and (at-robot ?from) (adjacent ?from ?to) (not (wall ?to)) (not (at-box ?to)))
    :effect (and (at-robot ?to) (not (at-robot ?from)))
  )

  (:action push
    :parameters (?from ?to ?boxloc - location)
    :precondition (and (at-robot ?from) (adjacent ?from ?to) (at-box ?to) (adjacent ?to ?boxloc) (not (wall ?boxloc)) (not (at-box ?boxloc)))
    :effect (and (at-robot ?to) (at-box ?boxloc) (not (at-robot ?from)) (not (at-box ?to)))
  )
)
