(define (problem sokoban-problem-variant)
  (:domain sokoban)
  (:objects
    l1 l2 l3 l4 l5 l6 - location
  )

  (:init
    (at-robot l1)        ; start
    (at-box l2)          ; first box
    (at-box l3)          ; seckond box
    (goal l5)            ; end loc for first
    (goal l6)            ; end loc for seckond
    (adjacent l1 l2)
    (adjacent l2 l1)
    (adjacent l2 l3)
    (adjacent l3 l2)
    (adjacent l3 l4)
    (adjacent l4 l3)
    (adjacent l4 l5)
    (adjacent l5 l4)
    (adjacent l5 l6)
    (adjacent l6 l5)
  )

  (:goal
    (and
      (at-box l5)        
      (at-box l6)        
    )
  )
)
